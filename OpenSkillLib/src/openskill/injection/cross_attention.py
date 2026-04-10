"""
cross_attention.py — S-Path-RAG Gap 3: Cross-Attention Injection (Eq 6)
========================================================================
Implements Eq 6 from the paper:

    Attn(Q_tok, K_graph, V_graph) = softmax(Q_tok @ K_graph.T / sqrt(d)) @ V_graph

Where:
    Q_tok   = LLM token hidden states at the target layer
    K_graph = projection of skill vectors to key space
    V_graph = projection of skill vectors to value space

Difference from prefix injection (Gap 2):
    - Prefix injection: skills enter as extra tokens at the start of the sequence.
      The LLM may ignore them in deep layers where context is already consolidated.
    - Cross-attention injection: skills are injected directly into chosen layers
      via their own attention mechanism, ensuring each target layer "sees" 
      skill knowledge regardless of previous context.

Implementation:
    - CrossAttentionInjector: nn.Module with K_proj, V_proj, and per-layer gate
    - Installed via register_forward_hook in the last N layers of Qwen
    - Gate initialized at 0.0 → null effect in untrained inference
    - Gate grows with training → controlled effect (~0.1 to 0.3 in practice)
    - Removable hooks: .remove_hooks() → model returns to original state

Usage:
    injector = CrossAttentionInjector(hidden_size=2048, n_layers=7)
    injector.install(model)           # installs hooks
    injector.set_skill_context(kvecs) # defines K_graph and V_graph
    out = model.generate(...)         # generation with cross-attention
    injector.remove_hooks()           # cleans up hooks
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

log = structlog.get_logger()

INJECTOR_SAVE_NAME  = "cross_attn_injector.safetensors"
# Fraction of layers to receive cross-attention (last N)
INJECTION_LAYER_FRAC = 0.25   # 25% → Qwen3.5-2B: layers 21-27 (7 of 28)
MIN_INJECTION_LAYERS = 4


# ── Main Module ──────────────────────────────────────────────────────────

class CrossAttentionInjector(nn.Module):
    """
    Injects K_graph and V_graph via cross-attention in deep layers of Qwen.

    Eq 6: output += gate * Attn(Q_tok, K_graph, V_graph)

    Trainable parameters per instance:
        k_proj: Linear(embed_dim, hidden_size)  — projects skills → keys
        v_proj: Linear(embed_dim, hidden_size)  — projects skills → values
        gates:  [n_layers]  — scalar gate per layer, init=0.0

    Non-trainable parameters (inference context):
        _K_graph: [1, N_skills, hidden_size]  — precomputed keys
        _V_graph: [1, N_skills, hidden_size]  — precomputed values
        _alphas:  [N_skills]  — weights per skill (Eq 5)
    """

    def __init__(
        self,
        embed_dim: int = 384,        # dimension of MiniLM vectors
        hidden_size: int = 2048,     # Qwen hidden size
        n_layers: int = 7,           # how many layers receive cross-attention
        num_heads: int = 16,         # Qwen num_key_value_heads (for head_dim)
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads

        # Shared projectors across all layers
        self.k_proj = nn.Linear(embed_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(embed_dim, hidden_size, bias=False)

        # Per-layer gate — initialized at 0.0 (no initial effect)
        # Uses tanh gate: output *= tanh(gate) → bounded within (-1, 1)
        self.gates = nn.Parameter(torch.zeros(n_layers))

        # Inference context (non-trainable, defined in set_skill_context)
        self._K_graph: Optional[torch.Tensor] = None   # [1, N, H]
        self._V_graph: Optional[torch.Tensor] = None   # [1, N, H]
        self._alphas:  Optional[torch.Tensor] = None   # [N]
        self._hooks: list = []

        # Xavier initialization for K and V proj
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def set_skill_context(
        self,
        skill_vectors: list[np.ndarray],
        skill_alphas: Optional[list[float]] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Precomputes K_graph and V_graph from skill vectors.
        Must be called before each generation.

        Integrated Eq 5: K and V are scaled by alpha_p before attention.
        """
        if not skill_vectors:
            self._K_graph = None
            self._V_graph = None
            self._alphas  = None
            return

        # --- FIX: Added .to(dtype) at the end ---
        skills_t = torch.tensor(
            np.array(skill_vectors), dtype=torch.float32, device=device
        ).to(dtype)

        with torch.no_grad():
            # Projects to LLM space
            K = self.k_proj(skills_t)  # [N, hidden_size]
            V = self.v_proj(skills_t)  # [N, hidden_size]

            # --- FIX: ENERGY LIMITER (L2 NORMALIZATION) ---
            # Prevents norm from exploding to 23 billion. Forces size 
            # to match LLM biology (sqrt(hidden_size))
            K = F.normalize(K, p=2, dim=-1) * (self.hidden_size ** 0.5)
            V = F.normalize(V, p=2, dim=-1) * (self.hidden_size ** 0.5)
            # --------------------------------------------------------

            # Scales by alpha_p (Eq 5): more relevant skills have larger keys/values
            if skill_alphas and len(skill_alphas) == len(skill_vectors):
                alpha_t = torch.tensor(
                    skill_alphas, dtype=dtype, device=device
                ).unsqueeze(-1)  # [N, 1]
                K = K * alpha_t
                V = V * alpha_t

            self._K_graph = K.unsqueeze(0)  # [1, N, H]
            self._V_graph = V.unsqueeze(0)  # [1, N, H]

        log.debug(
            "cross_attn.context_set",
            n_skills=len(skill_vectors),
            has_alphas=skill_alphas is not None,
            K_norm=f"{float(K.norm()):.3f}",
        )

    def _cross_attn_output(
        self,
        hidden_states: torch.Tensor,  # [B, T, H]
        layer_idx: int,               # relative index (0 = first target layer)
    ) -> torch.Tensor:
        """
        Calculates Attn(Q_tok, K_graph, V_graph) for a layer.

        Eq 6: softmax(Q @ K_graph.T / sqrt(d)) @ V_graph
        Output: [B, T, H] — same shape as hidden_states
        """
        if self._K_graph is None or self._V_graph is None:
            return torch.zeros_like(hidden_states)

        B, T, H = hidden_states.shape
        K_g = self._K_graph.to(hidden_states.device, hidden_states.dtype)  # [1, N, H]
        V_g = self._V_graph.to(hidden_states.device, hidden_states.dtype)  # [1, N, H]

        N = K_g.shape[1]

        # Q: [B, T, H] → reshape for multi-head: [B, num_heads, T, head_dim]
        # K_g, V_g: [B, N, H] → [B, num_heads, N, head_dim]
        # (we use hidden_size // num_heads as head_dim)
        nH  = self.num_heads
        hD  = self.head_dim

        Q = hidden_states.reshape(B, T, nH, hD).transpose(1, 2)  # [B, nH, T, hD]
        K = K_g.expand(B, -1, -1).reshape(B, N, nH, hD).transpose(1, 2)  # [B, nH, N, hD]
        V = V_g.expand(B, -1, -1).reshape(B, N, nH, hD).transpose(1, 2)  # [B, nH, N, hD]

        # Scaled dot-product attention
        scale = hD ** -0.5
        attn_w = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)  # [B, nH, T, N]
        attn_out = attn_w @ V                                              # [B, nH, T, hD]

        # Back to [B, T, H]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, H)

        # Per-layer gate (tanh to limit amplitude)
        gate = torch.tanh(self.gates[layer_idx])

        return attn_out * gate

    def _make_hook(self, layer_idx: int):
        """Creates the forward hook for layer layer_idx."""

        def hook(module, input, output):
            # DecoderLayer output is a tuple: (hidden_states, *extras)
            # extras can be: attn_weights, present_kv (optional)
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Only injects if context is defined
            if self._K_graph is None:
                return output

            try:
                delta = self._cross_attn_output(hidden_states, layer_idx)
                new_hidden = hidden_states + delta

                if rest is not None:
                    return (new_hidden,) + rest
                return new_hidden

            except Exception as e:
                log.warning("cross_attn.hook_error", layer=layer_idx, error=str(e))
                return output

        return hook

    def install(self, model: nn.Module) -> None:
        """
        Installs hooks in the last n_layers of Qwen.
        Idempotent: removes old hooks before installing new ones.
        """
        self.remove_hooks()

        # Navigates to decoder layers
        layers = None
        for attr in ["model", "transformer"]:
            backbone = getattr(model, attr, None)
            if backbone is not None:
                layers = getattr(backbone, "layers", None)
                if layers is not None:
                    break

        if layers is None:
            log.error("cross_attn.install_failed", reason="cannot find decoder layers")
            return

        total = len(layers)
        start = max(0, total - self.n_layers)
        target_layers = list(range(start, total))

        for rel_idx, abs_idx in enumerate(target_layers):
            hook = layers[abs_idx].register_forward_hook(self._make_hook(rel_idx))
            self._hooks.append(hook)

        log.info(
            "cross_attn.installed",
            total_layers=total,
            injecting=target_layers,
            n_gates=self.n_layers,
        )

    def remove_hooks(self) -> None:
        """Removes all installed hooks. Model returns to original state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        log.debug("cross_attn.hooks_removed")

    def clear_context(self) -> None:
        """Clears skill context after generation."""
        self._K_graph = None
        self._V_graph = None
        self._alphas  = None

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file
        state = {k: v for k, v in self.state_dict().items()
                 if not k.startswith("_")}
        save_file(state, str(path))
        log.info("cross_attn.saved", path=str(path),
                 gates=[f"{float(g):.4f}" for g in self.gates.data])

    @classmethod
    def load(
        cls,
        path: str | Path,
        embed_dim: int,
        hidden_size: int,
        n_layers: int,
        num_heads: int,
        device: str = "cpu",
    ) -> "CrossAttentionInjector":
        from safetensors.torch import load_file
        inj = cls(embed_dim, hidden_size, n_layers, num_heads)
        inj.load_state_dict(load_file(str(path)), strict=False)
        inj.to(device)
        log.info("cross_attn.loaded", path=str(path),
                 gates=[f"{float(g):.4f}" for g in inj.gates.data])
        return inj

    @property
    def is_trained(self) -> bool:
        """True if gates diverged from zero."""
        return float(self.gates.abs().max()) > 1e-4


# ── Helper: automatically detects Qwen parameters ───────────────────────

def detect_qwen_params(model: nn.Module) -> dict:
    """
    Detects hidden_size, num_heads, and n_layers from model.config.
    Returns dict compatible with CrossAttentionInjector.__init__.
    """
    cfg = model.config
    hidden_size = cfg.hidden_size

    # Qwen2/Qwen3 uses num_key_value_heads for GQA
    # For cross-attn we use num_attention_heads (full heads)
    num_heads = getattr(cfg, "num_attention_heads", hidden_size // 128)

    total_layers = cfg.num_hidden_layers
    n_inj = max(MIN_INJECTION_LAYERS,
                int(total_layers * INJECTION_LAYER_FRAC))

    return {
        "hidden_size": hidden_size,
        "num_heads":   num_heads,
        "n_layers":    n_inj,
    }
