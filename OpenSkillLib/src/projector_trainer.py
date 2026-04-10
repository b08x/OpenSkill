"""
projector_trainer.py — SkillProjector training with L_align (Gap 2)
====================================================================
Trains the SkillProjector so the LLM truly attends to the injected vectors.

Eq 7: attnMass(p) = (1/T_tok) * Σ_t Σ_{k ∈ idx(p)} A_{t,k}
Eq 8: L_align = (1/|P_sel|) * Σ_p (alpha_p - attnMass_p)²

Pipeline:
  1. Loads already trained scorer (path_scorer.safetensors)
  2. Loads LLM (Qwen) — frozen, no gradients
  3. Loads skills + embeddings from store
  4. For each query in the dataset:
     a. Scorer calculates alpha_p per skill (Eq 5)
     b. Projector injects skills as prefix tokens
     c. Forward pass with output_attentions=True
     d. L_align = (alpha_p - attnMass_p)²
     e. Backward only on the projector (frozen LLM)
  5. Saves projector_weights.safetensors

Usage:
    python projector_trainer.py --skill-dir ./skills_output --model-id Qwen/Qwen2.5-0.5B

    # Use existing dataset
    python projector_trainer.py --skill-dir ./skills_output --data-cache train_data.npz

    # Adjust epochs and LR
    python projector_trainer.py --skill-dir ./skills_output --epochs 30 --lr 5e-4
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import structlog

log = structlog.get_logger()

TrainSample = Tuple[np.ndarray, np.ndarray, bool]

PROJECTOR_SAVE_NAME  = "projector_weights.safetensors"
SCORER_SAVE_NAME     = "path_scorer.safetensors"
DEFAULT_MODEL_ID     = "Qwen/Qwen3.5-2B"
BATCH_SIZE           = 4    # small — each sample requires an LLM forward pass
LAMBDA_ALIGN         = 0.5  # L_align weight in total loss


# ── Loading Frozen LLM ─────────────────────────────────────────────

def load_frozen_llm(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"  Loading LLM ({model_id}) — frozen...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Let transformers choose ideal dtype (usually BF16 for Qwen)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    return model, tokenizer


# ── Collecting alpha_p via scorer ──────────────────────────────────────────────

def compute_alphas_for_sample(
    scorer,
    query_vec: np.ndarray,
    skill_vecs: list[np.ndarray],
    device: str,
) -> list[float]:
    """
    Runs the scorer for each skill and returns alpha_p = softmax(u_p) * v_eta_p,
    normalized (Eq 4 × Eq 5).
    """
    if not skill_vecs:
        return []

    q_t = torch.tensor(query_vec, dtype=torch.float32, device=device)
    us, vetas = [], []

    with torch.no_grad():
        for sv in skill_vecs:
            q_in = q_t.unsqueeze(0)
            p_in = torch.tensor(sv, dtype=torch.float32, device=device).unsqueeze(0)
            u, v = scorer(q_in, p_in)
            us.append(u.squeeze())
            vetas.append(v.squeeze())

    u_t     = torch.stack(us)                    # [N]
    v_t     = torch.stack(vetas)                 # [N]
    w_tilde = torch.softmax(u_t, dim=0)          # [N]
    alpha   = w_tilde * v_t                      # [N]
    alpha   = alpha / (alpha.sum() + 1e-8)       # normalize

    return alpha.tolist()


# ── Forward pass for L_align ─────────────────────────────────────────────────

def compute_lalign_for_batch(
    model,
    tokenizer,
    projector,
    scorer,
    queries: list[str],
    skill_vecs_per_query: list[list[np.ndarray]],
    alphas_per_query: list[list[float]],
    device: str,
) -> torch.Tensor:
    """
    Calculates L_align for a batch of queries.

    For each query:
      1. Tokenizes the prompt
      2. Projects skills → prefix tokens (with gradient on projector)
      3. Forward pass with output_attentions=True
      4. Extracts attnMass for each skill
      5. L_align = mean((alpha_p - attnMass_p)²)

    Returns the mean L_align over the batch.
    """
    from openskill.injection.soft import capture_attn_mass, create_injected_attention_mask

    batch_align_losses = []

    for query, skill_vecs, alphas in zip(queries, skill_vecs_per_query, alphas_per_query):
        if not skill_vecs or not alphas:
            continue

        N = len(skill_vecs)

        # Tokenize prompt (no input_ids in generate — we use embeds)
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            # Embedding of prompt tokens [1, T, D_llm]
            token_embeds = model.get_input_embeddings()(enc.input_ids)

        # Projects skills → [N, D_skill] → [N, D_llm]  (WITH gradient)
        skills_t = torch.tensor(
            np.array(skill_vecs),
            dtype=torch.float32,  # Keep input in float32 for projector
            device=device,
        )
        projected = projector(skills_t)  # Output is Float32 here

        projected = projected.to(model.dtype)

        # Energy scaling
        with torch.no_grad():
            token_embeds = model.get_input_embeddings()(enc.input_ids)
            prompt_norm = token_embeds.norm(p=2, dim=-1).mean()

        proj_norm = projected.norm(p=2, dim=-1).mean()
        projected_scaled = projected * (prompt_norm / (proj_norm + 1e-8))

        # concat will work now because both are model.dtype (BFloat16)
        combined = torch.cat(
            [projected_scaled.unsqueeze(0), token_embeds], dim=1
        )

        # Expanded attention mask [1, N + T]
        combined_mask = create_injected_attention_mask(
            enc.attention_mask, N, device
        )

        # Capture attnMass — l_align has gradient via combined (depends on projector)
        l_align, attn_mass = capture_attn_mass(
            model=model,
            input_embeds=combined,
            attention_mask=combined_mask,
            n_skill_tokens=N,
            skill_alphas=alphas,
            device=device,
        )

        if l_align.requires_grad:
            batch_align_losses.append(l_align)

        log.debug(
            "projector.lalign_sample",
            query=query[:40],
            alphas=[f"{a:.3f}" for a in alphas],
            attn_mass=[f"{float(m):.3f}" for m in attn_mass.tolist()],
            l_align=f"{float(l_align):.4f}",
        )

    if not batch_align_losses:
        return torch.tensor(0.0, requires_grad=True, device=device)

    return torch.stack(batch_align_losses).mean()


# ── Projector Training Loop ───────────────────────────────────────────────

async def train_projector(
    skill_dir: str,
    model_id: str,
    data_cache: str,
    epochs: int = 30,
    lr: float = 5e-4,
    lambda_align: float = LAMBDA_ALIGN,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skill_path = Path(skill_dir)

    # ── 1. Check Scorer ────────────────────────────────────────────────────
    scorer_path = skill_path / SCORER_SAVE_NAME
    if not scorer_path.exists():
        print(f"\nERROR: {scorer_path} not found.")
        print("Run first: python train_scorer.py --skill-dir", skill_dir)
        sys.exit(1)

    from safetensors.torch import load_file as st_load
    from openskill.core.trainer import PathScorerModel
    from openskill.injection.soft import SkillProjector

    scorer = PathScorerModel(embed_dim=384).to(device)
    scorer.load_state_dict(st_load(str(scorer_path)))
    scorer.eval()
    print(f"  Scorer loaded: {scorer_path}")

    # ── 2. Load Frozen LLM ──────────────────────────────────────────────
    model, tokenizer = load_frozen_llm(model_id, device)
    hidden_size = model.config.hidden_size

    # ── 3. Initialize Projector ───────────────────────────────────────────────
    projector_path = skill_path / PROJECTOR_SAVE_NAME
    if projector_path.exists():
        projector = SkillProjector.load(
            projector_path, embed_dim=384, llm_hidden_size=hidden_size, device=device
        )
        projector = projector.to(torch.float32)
        print(f"  Existing projector loaded: {projector_path}")
    else:
        projector = SkillProjector(embed_dim=384, llm_hidden_size=hidden_size)
        # Initialize with small perturbation around zero
        # (NOT eye_ — eye_ only works for square matrices and is suboptimal here)
        torch.nn.init.xavier_uniform_(projector.proj.weight)
        torch.nn.init.zeros_(projector.proj.bias)
        projector = projector.to(device).to(torch.float32)
        print(f"  New projector initialized (Xavier): {hidden_size}d")

    projector.train()

    # ── 4. Load Dataset ────────────────────────────────────────────────────
    cache = Path(data_cache)
    if cache.exists():
        from bootstrap_data import load_dataset
        train_data, val_data = load_dataset(str(cache))
    else:
        print(f"\n  Dataset not found in {cache}.")
        print("  Run first: python train_scorer.py --skill-dir", skill_dir, "--only-data")
        sys.exit(1)

    # Filter only positives — only queries with correct answers
    # are useful for L_align (we want LLM to attend to the correct skill)
    positive_samples = [(q, p) for q, p, is_pos in train_data if is_pos]
    print(f"\n  {len(positive_samples)} positive samples for projector training")

    if len(positive_samples) == 0:
        print("ERROR: No positive samples in the dataset.")
        sys.exit(1)

    # Load skills from store — keep (title, vec) for representative queries
    from openskill.storage.local import LocalDiskStore
    store = LocalDiskStore(skill_dir)
    all_metas = await store.list_skills()

    skill_index: list[tuple[str, np.ndarray]] = []
    for m in all_metas:
        title = getattr(m, 'title', '') or ''
        vectors = getattr(m, 'vectors', {})
        for p in vectors.values():
            if getattr(p, 'dimension', 0) == 384 and p.embedding:
                skill_index.append((title, np.array(p.embedding, dtype=np.float32)))
                break

    if not skill_index:
        print("ERROR: No skills with 384d embedding found.")
        print("Run: openskill embed --local --skill-id <id>")
        sys.exit(1)

    skill_vecs_store = [v for _, v in skill_index]
    skill_titles     = [t for t, _ in skill_index]
    print(f"  {len(skill_vecs_store)} skills with 384d vectors available")

    # ── 5. Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)

    # LR scheduler: cosine with warmup
    import math
    steps_per_epoch = math.ceil(len(positive_samples) / BATCH_SIZE)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # ── 6. Training Loop ─────────────────────────────────────────────────────
    print(f"\n  Training projector for {epochs} epochs...")
    print(f"  LR: {lr}  |  Lambda_align: {lambda_align}  |  Batch: {BATCH_SIZE}")
    print(f"  Attention layers used: last {int(model.config.num_hidden_layers * 0.5)}")
    print()

    best_loss = float("inf")
    import random

    for epoch in range(epochs):
        random.shuffle(positive_samples)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(positive_samples), BATCH_SIZE):
            batch = positive_samples[start:start + BATCH_SIZE]
            optimizer.zero_grad()

            # Uses skill title as representative query instead of "query_N"
            # This anchors the LLM prompt to the actual semantic domain of the skill
            batch_skill_vecs = []
            batch_alphas     = []
            queries          = []

            for q_vec, s_vec in batch:
                # Find skill closest to q_vec to name the query
                sims = [float(np.dot(q_vec, sv) /
                              (np.linalg.norm(q_vec) * np.linalg.norm(sv) + 1e-8))
                        for sv in skill_vecs_store]
                best_idx = int(np.argmax(sims))
                query_text = f"How to implement {skill_titles[best_idx]}?" \
                             if skill_titles[best_idx] else "How to solve this technical problem?"
                queries.append(query_text)

                # With multiple skills: include correct skill + 1 random negative
                # This trains the projector on the mixture, not just isolated skills
                if len(skill_vecs_store) > 1:
                    neg_candidates = [i for i in range(len(skill_vecs_store))
                                      if not np.allclose(skill_vecs_store[i], s_vec, atol=1e-4)]
                    if neg_candidates:
                        neg_idx = random.choice(neg_candidates)
                        path_vecs = [s_vec, skill_vecs_store[neg_idx]]
                    else:
                        path_vecs = [s_vec]
                else:
                    path_vecs = [s_vec]

                alphas = compute_alphas_for_sample(scorer, q_vec, path_vecs, device)
                batch_skill_vecs.append(path_vecs)
                batch_alphas.append(alphas)

            # Calculate L_align for the batch
            l_align = compute_lalign_for_batch(
                model=model,
                tokenizer=tokenizer,
                projector=projector,
                scorer=scorer,
                queries=queries,
                skill_vecs_per_query=batch_skill_vecs,
                alphas_per_query=batch_alphas,
                device=device,
            )

            # L2 regularization on projector weights (prevents divergence)
            l_reg = sum(p.norm() ** 2 for p in projector.parameters()) * 1e-5

            loss = lambda_align * l_align + l_reg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if n_batches == 0:
            continue

        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            projector.save(str(projector_path))

        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(
                "projector.epoch",
                epoch=epoch,
                loss=f"{avg_loss:.4f}",
                best=f"{best_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

    # ── 7. Final Verification ──────────────────────────────────────────────────
    print(f"\n  Best L_align: {best_loss:.4f}")
    print(f"  Projector saved to: {projector_path}")

    # Confirm projector diverged from eye_
    proj = SkillProjector.load(
        projector_path, embed_dim=384, llm_hidden_size=hidden_size, device=device
    )
    print(f"  Projector trained: {proj.is_trained}")

    return {"best_loss": best_loss, "path": str(projector_path)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Trains SkillProjector with L_align (S-Path-RAG Gap 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skill-dir",  default="./skills_output")
    parser.add_argument("--model-id",   default=DEFAULT_MODEL_ID,
                        help=f"Qwen model (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--data-cache", default="train_data.npz")
    parser.add_argument("--regen-data", action="store_true",
                        help="Regenerates dataset even if cache exists (use when adding new skills)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--lambda-align", type=float, default=LAMBDA_ALIGN,
                        help="L_align weight in loss (default: 0.5)")
    args = parser.parse_args()

    if not Path(args.skill_dir).exists():
        print(f"ERROR: '{args.skill_dir}' not found.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  SkillProjector Training — L_align (S-Path-RAG Eq 7+8)")
    print("=" * 60)

    # Warning when old cache exists and regen not requested
    cache = Path(args.data_cache)
    if cache.exists() and not args.regen_data:
        print(f"\n  WARNING: Using cached dataset '{args.data_cache}'.")
        print("  If you added new skills, use --regen-data to include them.")
        print()

    asyncio.run(train_projector(
        skill_dir    = args.skill_dir,
        model_id     = args.model_id,
        data_cache   = args.data_cache if not args.regen_data else "__force_regen__",
        epochs       = args.epochs,
        lr           = args.lr,
        lambda_align = args.lambda_align,
    ))

    print("\n  Next steps:")
    print("  1. local_llm.py loads the projector automatically")
    print("  2. Use --mode injection to activate trained injection")
    print("  3. Run: openskill retrieve --local --mode injection --query 'your query'")


if __name__ == "__main__":
    main()