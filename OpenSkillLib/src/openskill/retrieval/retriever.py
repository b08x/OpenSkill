"""
retriever.py — OpenSkillRetriever: Unified S-Path-RAG & TurboQuant Retrieval
=============================================================================
Changes from previous version:

  - RetrievalGuidance.skill_alphas: list[float]  (NEW — Eq 5)
  - retrieve(): propagates best_path.node_scores → guidance.skill_alphas
  - aggregate_path_vectors(): uses real skill_alphas (Eq 5) with uniform fallback

No public interface was broken.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING

import structlog
from openskill.core.vector import TurboQuantizer, unpack_qvector
from openskill.core.graph import SkillGraph, GraphRetrievalResult

if TYPE_CHECKING:
    from openskill.storage.base import BaseSkillStore, SkillMetadata
    from openskill.llm.base import BaseLLMProvider

log = structlog.get_logger()


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RetrievalGuidance:
    """The complete guidance package for the Generator LLM."""
    query: str
    best_path_ids: list[str]

    # Dequantized vectors ready for Soft Latent Injection
    skill_vectors: list[np.ndarray] = field(default_factory=list)

    # Eq 5: alpha_p = w_tilde_p * v_eta_p, normalized
    # One float per skill in best_path_ids (same order as skill_vectors)
    skill_alphas: list[float] = field(default_factory=list)

    # Markdown content of skills in path
    skill_contents: list[str] = field(default_factory=list)

    # Detailed metadata
    skills_meta: list[dict] = field(default_factory=list)

    # Socratic process diagnostics
    confidence: float = 0.0
    reasoning_trace: str = ""


# ── Main Class ──────────────────────────────────────────────────────────

class OpenSkillRetriever:
    def __init__(
        self,
        store: BaseSkillStore,
        llm: BaseLLMProvider,
        quantizer: Optional[TurboQuantizer] = None,
    ):
        self.store = store
        self.llm = llm
        self._quantizer_override = quantizer
        self.quantizer = quantizer or TurboQuantizer(dimension=384)
        self.graph = SkillGraph(store=self.store)

    async def retrieve(
            self,
            query: str,
            top_k: int = 3,
            use_graph: bool = True,
            use_gnn: bool = False
    ) -> RetrievalGuidance:
        log.info("retrieval.start", query=query[:50])

        # 1. Get Query Vec
        query_vec = await self.quantizer.embed(self.llm, query)

        # 2. Hybrid Search (Graph)
        graph_result = await self.graph.find_paths(
            query_vec=query_vec,
            query_text=query,
            top_k=top_k,
            use_gnn=use_gnn
        )

        # 3. Intelligent Fallback: If Graph failed, search by simple vector similarity
        skill_ids_to_process = graph_result.paths[0].node_ids if graph_result.paths else []

        if not skill_ids_to_process:
            log.info("retrieval.fallback_to_semantic_search")
            all_metas = await self.store.list_skills()
            # Sort skills by simple vector similarity
            sims = []
            for m in all_metas:
                # Extract skill vector (same logic as _get_seeds)
                vec = None
                if m.vectors and 'default' in m.vectors:
                    vec = np.array(m.vectors['default'].embedding)
                elif m.embedding:
                    vec = np.array(m.embedding)

                if vec is not None:
                    sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
                    sims.append((m.id, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            skill_ids_to_process = [s[0] for s in sims[:top_k]]

        # 4. Assemble guidance (now with guaranteed skill_ids_to_process)
        # Use node_scores from scorer (Eq 5) if available, otherwise uniform fallback
        best_node_scores = (
            graph_result.paths[0].node_scores
            if graph_result.paths and graph_result.paths[0].node_scores
            else []
        )
        n_ids = len(skill_ids_to_process)
        if best_node_scores and len(best_node_scores) == n_ids:
            skill_alphas = best_node_scores
        else:
            skill_alphas = [1.0 / n_ids] * n_ids if n_ids else []

        guidance = RetrievalGuidance(
            query=query,
            best_path_ids=skill_ids_to_process,
            confidence=graph_result.confidence,
            reasoning_trace=graph_result.trace,
            skill_alphas=skill_alphas,
        )

        # 5. Enrichment (Same as always)
        for skill_id in skill_ids_to_process:
            meta = await self.store.get_skill_meta(skill_id)
            if not meta: continue

            correct_qv_dict = None
            query_dim = query_vec.shape[0]

            if meta.vectors:
                gnn_profile, fallback_profile = None, None
                for profile in meta.vectors.values():
                    if profile.dimension == query_dim:
                        if getattr(profile, 'provider', '') == "OpenSkillGNN":
                            gnn_profile = profile
                        else:
                            fallback_profile = profile

                chosen = (gnn_profile if (use_gnn and gnn_profile) else None) \
                         or fallback_profile or gnn_profile
                if chosen:
                    correct_qv_dict = chosen.qvector

            if not correct_qv_dict and getattr(meta, 'embedding_dim', 0) == query_dim:
                correct_qv_dict = meta.qvector

            if correct_qv_dict and isinstance(correct_qv_dict, dict) and "qvec" in correct_qv_dict:
                try:
                    qv = unpack_qvector(correct_qv_dict)
                    v_approx = self.quantizer.dequantize(qv)
                    guidance.skill_vectors.append(v_approx)
                except Exception as e:
                    log.warning("retriever.dequantize_failed", skill_id=skill_id, error=str(e))

            content = await self.store.get_skill_md(skill_id)
            guidance.skill_contents.append(content or "")
            guidance.skills_meta.append(meta.to_dict())

        log.info(
            "retrieval.done",
            path_len=len(guidance.best_path_ids),
            confidence=f"{guidance.confidence:.2f}",
            has_alphas=len(guidance.skill_alphas) > 0,
            alphas=[f"{a:.3f}" for a in guidance.skill_alphas],
        )

        return guidance

    async def get_raw_similarity(self, query: str, skill_id: str) -> float:
        query_vec = await self.quantizer.embed(self.llm, query)
        q_qv = self.quantizer.quantize(query_vec)
        meta = await self.store.get_skill_meta(skill_id)
        if not meta or not meta.qvector:
            return 0.0
        skill_qv = unpack_qvector(meta.qvector)
        return self.quantizer.similarity(q_qv, skill_qv)


# ── Eq 5: Compact Soft Mixture ────────────────────────────────────────────────

def aggregate_path_vectors(guidance: RetrievalGuidance) -> np.ndarray:
    """
    Eq 5 (S-Path-RAG): z_ctx = Σ alpha_p · Encpath(p)

    alpha_p = w_tilde_p * v_eta_p, normalized (calculated in graph._score_path)

    If skill_alphas is empty (scorer not loaded), uses uniform fallback
    to not break verbalization mode.
    """
    if not guidance.skill_vectors:
        return np.zeros(384)

    vectors = np.array(guidance.skill_vectors)   # [N, D]
    N = len(vectors)

    # ── Neural path: real alphas from scorer ────────────────────────────────
    if guidance.skill_alphas and len(guidance.skill_alphas) == N:
        alphas = np.array(guidance.skill_alphas, dtype=np.float32)

        # Ensures sum = 1 (normalized mixture coefficients)
        alphas = alphas / (alphas.sum() + 1e-8)

        z_ctx = np.average(vectors, axis=0, weights=alphas)

        log.debug(
            "aggregate.neural",
            n=N,
            alphas=[f"{a:.3f}" for a in alphas],
        )

    # ── Fallback: partial alphas (path longer than available vectors)
    elif guidance.skill_alphas:
        n_avail = min(N, len(guidance.skill_alphas))
        alphas = np.array(guidance.skill_alphas[:n_avail], dtype=np.float32)
        alphas = alphas / (alphas.sum() + 1e-8)
        z_ctx = np.average(vectors[:n_avail], axis=0, weights=alphas)
        log.debug("aggregate.partial_alphas", n_used=n_avail, n_total=N)

    # ── Uniform fallback: untrained scorer or manual mode ─────────────────
    else:
        z_ctx = np.mean(vectors, axis=0)
        log.debug("aggregate.uniform_fallback", n=N)

    # Normalizes to maintain latent space scale
    norm = np.linalg.norm(z_ctx) + 1e-8
    return z_ctx / norm
