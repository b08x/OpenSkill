"""
graph.py — S-Path-RAG: Semantic Skill Graph Retrieval
=====================================================
Changes from previous version:

  - SkillPath.node_scores: list[float]  (NEW — Eq 4 × Eq 5)
  - _score_path neural mode: calculates alpha_p = w_tilde_p * v_eta_p per node
  - _score_path stores result in self._last_node_scores for find_paths to read
  - find_paths: populates SkillPath.node_scores with calculated alphas

No public interface was broken.
"""

from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING
import structlog
import torch
from safetensors.torch import load_file

from openskill.core.gnn_encoder import encode_graph_embeddings

if TYPE_CHECKING:
    from openskill.storage.base import BaseSkillStore

log = structlog.get_logger()

MAX_PATH_LEN = 4
BEAM_WIDTH = 8
CONFIDENCE_THRESHOLD = 0.35
ALPHA, BETA, GAMMA = 0.4, 0.4, 0.2

EDGE_WEIGHTS = {
    "PREREQUISITE_OF": 0.9, "EXTENDS": 0.85, "RESOLVES_ERROR": 0.8,
    "SIMILAR_TO": 0.7, "CONTRADICTS": -0.5
}


@dataclass
class SkillPath:
    node_ids: list[str]
    score: float = 0.0
    # Eq 4 × Eq 5: alpha_p = softmax(u_p) * v_eta_p, normalized per node
    # Used by aggregate_path_vectors for true weighted mixture
    node_scores: list[float] = field(default_factory=list)


@dataclass
class GraphRetrievalResult:
    paths: list[SkillPath]
    confidence: float
    rounds: int
    trace: str


class SkillGraph:
    def __init__(self, store: BaseSkillStore):
        self.store = store
        self._neural_scorer = None
        self._last_node_scores: list[float] = []   # cache between _score_path and find_paths
        self._load_scorer()

    def _load_scorer(self):
        """Attempts to load the trained scorer from disk (auto-detects dimension)."""
        from openskill.core.trainer import PathScorerModel
        import safetensors.torch
        try:
            model_path = self.store.workspace_path / "path_scorer.safetensors"
            if model_path.exists():
                # 1. Discover real dimension by reading 1st layer weights (trunk.0.weight)
                with safetensors.safe_open(model_path, framework="pt") as f:
                    tensor_shape = f.get_tensor("trunk.0.weight").shape
                    # Input is embed_dim * 2, so real dimension is shape[1] / 2
                    real_dim = tensor_shape[1] // 2

                # 2. Load the model with correct dimension
                self._neural_scorer = PathScorerModel(embed_dim=real_dim)
                self._neural_scorer.load_state_dict(safetensors.torch.load_file(model_path))
                self._neural_scorer.eval()
                log.info("graph.neural_scorer_loaded", dim=real_dim)
        except Exception as e:
            log.warning("graph.scorer_load_failed", error=str(e))

    async def find_paths(
            self,
            query_vec: np.ndarray,
            query_text: str, # NEW: Receives original text
            top_k: int = 3,
            use_gnn: bool = False
    ) -> GraphRetrievalResult:
        graph_data = self.store.get_graph()
        adj = self._build_adjacency(graph_data.edges)
        all_metas = await self.store.list_skills()

        seeds = self._get_seeds(query_vec, query_text, all_metas, top_k=5, use_gnn=use_gnn)

        rounds, best_paths, confidence, socratic_trace = 0, [], 0.0, []

        while rounds < 3:
            rounds += 1
            candidate_paths = self._enumerate_paths(seeds, adj, max_len=MAX_PATH_LEN + rounds)
            scored_paths = []

            for path_ids in candidate_paths:
                self._last_node_scores = []
                score = self._score_path(
                    path_ids, query_vec, query_text, all_metas, graph_data.edges, use_gnn=use_gnn
                )
                node_sc = list(self._last_node_scores)
                scored_paths.append(
                    SkillPath(node_ids=path_ids, score=score, node_scores=node_sc)
                )

            scored_paths.sort(key=lambda x: x.score, reverse=True)
            best_paths = scored_paths[:top_k]
            if not best_paths:
                break

            confidence = best_paths[0].score
            socratic_trace.append(f"Round {rounds}: Top Score {confidence:.2f}")
            if confidence >= CONFIDENCE_THRESHOLD:
                break
            socratic_trace.append("Confidence low. Expanding graph search...")

        return GraphRetrievalResult(best_paths, confidence, rounds, "\n".join(socratic_trace))

    def _get_seeds(self, query_vec: np.ndarray, query_text: str, metas: list[Any], top_k: int, use_gnn: bool = False) -> \
    list[str]:
        query_dim = query_vec.shape[0]
        scores = []

        stop_words = {"the", "find", "calculate", "how", "to", "in", "of", "a", "and", "is", "for", "sequence", "using",
                      "with"}
        query_keywords = set(
            word.lower().strip("?!.,") for word in query_text.split() if word.lower() not in stop_words)

        for m in metas:
            target_vec = None
            # Ensures access via attribute (SkillMetadata class)
            skill_id = getattr(m, 'id', 'unknown')
            title = getattr(m, 'title', '').lower()

            # DEBUG: Now using getattr to avoid AttributeError
            print(f"[DEBUG RETRIEVAL] Analyzing skill: {getattr(m, 'title', 'no title')}")
            print(f"  Query keywords: {query_keywords}")
            vectors = getattr(m, 'vectors', {})

            if isinstance(vectors, dict):
                for profile in vectors.values():
                    # Safe access for dimension and embedding
                    p_dim = getattr(profile, 'dimension', 0) if not isinstance(profile, dict) else profile.get(
                        'dimension', 0)
                    p_emb = getattr(profile, 'embedding', None) if not isinstance(profile, dict) else profile.get(
                        'embedding', None)

                    if p_dim == query_dim and p_emb is not None:
                        target_vec = np.array(p_emb)
                        break

            # Root fallback
            if target_vec is None:
                m_emb = getattr(m, 'embedding', None)
                if m_emb is not None and len(m_emb) == query_dim:
                    target_vec = np.array(m_emb)

            # --- Hybrid Score ---
            if target_vec is not None:
                norm_q = np.linalg.norm(query_vec) + 1e-8
                norm_t = np.linalg.norm(target_vec) + 1e-8
                vector_sim = np.dot(query_vec, target_vec) / (norm_q * norm_t)

                lexical_sim = 0.0
                if title and query_keywords:
                    title_words = set(title.replace("(", "").replace(")", "").lower().split())
                    overlap = query_keywords.intersection(title_words)
                    if overlap:
                        lexical_sim = (len(overlap) / len(query_keywords)) * 0.8

                combined_seed_score = (vector_sim * 0.4) + (lexical_sim * 0.6)
                scores.append((skill_id, combined_seed_score))
            else:
                # Optional debug log if you want to see which skill lacks a vector
                # log.debug("graph.seed_no_matching_dim", skill=skill_id)
                pass
        # debug removed — m is SkillMetadata, not dict
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:top_k] if s[0]]

    def _score_path(self, path_ids, query_vec, query_text, metas, all_edges, use_gnn=False) -> float:
        query_dim = query_vec.shape[0]
        meta_map = {getattr(m, 'id', None): m for m in metas
                    if getattr(m, 'id', None)}

        path_vecs, path_sims, categories = [], [], set()
        lexical_bonus = 0.0

        stop_words = {"the", "find", "calculate", "how", "to", "in", "of", "a", "and", "is", "for"}
        query_keywords = set(
            word.lower().strip("?!.,") for word in query_text.split() if word.lower() not in stop_words)

        for nid in path_ids:
            m = meta_map.get(nid)
            if not m: continue

            # Lexical Bonus in Path Score
            title = (getattr(m, 'title', '') or '').lower()
            if title and query_keywords:
                title_words = set(title.lower().split())
                overlap = query_keywords.intersection(title_words)
                if overlap:
                    # Increased lexical bonus to 0.4
                    lexical_bonus += (len(overlap) / len(query_keywords)) * 0.4

            # Vector extraction same as _get_seeds
            target_emb = None
            vectors_data = getattr(m, 'vectors', {})
            if isinstance(vectors_data, dict):
                for p in vectors_data.values():
                    if getattr(p, 'dimension', 0) == query_dim:
                        target_emb = p.embedding
                        break

            if target_emb is not None:
                emb_arr = np.array(target_emb)
                path_vecs.append(emb_arr)
                norm_q = np.linalg.norm(query_vec) + 1e-8
                norm_t = np.linalg.norm(emb_arr) + 1e-8
                path_sims.append(np.dot(query_vec, emb_arr) / (norm_q * norm_t))
                categories.add(getattr(m, 'category', 'General'))

        if not path_vecs:
            return 0.0

        # ── NEURAL MODE: Eq 3 + Eq 4 + alpha_p ──────────────────────────────
        if self._neural_scorer is not None and query_dim == self._neural_scorer.embed_dim:
            with torch.no_grad():
                device = next(self._neural_scorer.parameters()).device
                q_t = torch.tensor(query_vec, dtype=torch.float32).to(device)

                # Scorer and verifier PER NODE (not path average)
                per_node_u = []  # u_p: raw score (Eq 3)
                per_node_veta = []  # v_eta(p,q): verifier ∈ (0,1)

                for pv in path_vecs:
                    q_in = q_t.unsqueeze(0)
                    p_in = torch.tensor(pv, dtype=torch.float32).unsqueeze(0).to(device)
                    s, prob = self._neural_scorer(q_in, p_in)
                    per_node_u.append(s.squeeze())
                    per_node_veta.append(prob.squeeze())

                u = torch.stack(per_node_u)  # [N]
                v_eta = torch.stack(per_node_veta)  # [N]

                # Eq 4: w_tilde_p = softmax(u_p / tau), tau=1.0 at inference
                w_tilde = torch.softmax(u, dim=0)  # [N]

                # Eq 5: alpha_p ∝ w_tilde_p * v_eta_p
                alpha = w_tilde * v_eta  # [N], not normalized
                alpha_norm = alpha / (alpha.sum() + 1e-8)  # normalize → mixture coeffs

                # Propagates alphas to find_paths via self._last_node_scores
                self._last_node_scores = alpha_norm.cpu().tolist()

                log.debug(
                    "graph.node_alphas",
                    path=path_ids,
                    alphas=[f"{a:.3f}" for a in self._last_node_scores],
                )

                # Global path score = sigmoid(u_mean) * v_eta_mean
                path_score = float(torch.sigmoid(u.mean()) * v_eta.mean())
                return path_score

        # ── MANUAL MODE (fallback without scorer) ────────────────────────────────
        # node_scores empty → aggregate_path_vectors will use uniform fallback
        self._last_node_scores = []
        semantic = np.mean(path_sims) if path_sims else 0.0

        edge_weights = []
        for i in range(len(path_ids) - 1):
            w = 0.5
            for e in all_edges:
                if e['from'] == path_ids[i] and e['to'] == path_ids[i + 1]:
                    w = EDGE_WEIGHTS.get(e['type'], 0.5) * e.get('weight', 1.0)
                    break
            edge_weights.append(w)

        structural = np.mean(edge_weights) if edge_weights else 1.0
        diversity = min(len(categories) / 3.0, 1.0)

        # Final score now adds lexical bonus and caps at 1.0
        final_score = (0.2 * structural) + (0.3 * semantic) + (0.1 * diversity) + lexical_bonus
        return float(min(final_score, 1.0))

    def _build_adjacency(self, edges: list[dict]) -> dict[str, list[dict]]:
        adj = defaultdict(list)
        for e in edges:
            adj[e['from']].append(e)
        return adj

    def _enumerate_paths(self, seeds: list[str], adj: dict[str, list[dict]], max_len: int) -> list[list[str]]:
        paths = []
        queue = deque([(s, [s]) for s in seeds])
        while queue:
            curr_id, path_so_far = queue.popleft()
            if len(path_so_far) >= max_len:
                paths.append(path_so_far)
                continue
            neighbors = adj.get(curr_id, [])
            if not neighbors:
                paths.append(path_so_far)
                continue
            neighbors = sorted(neighbors, key=lambda x: x.get('weight', 0.5), reverse=True)[:BEAM_WIDTH]
            for edge in neighbors:
                next_node = edge['to']
                if next_node not in path_so_far:
                    queue.append((next_node, path_so_far + [next_node]))
        return paths

    async def get_embeddings(self, use_gnn: bool = False) -> dict[str, np.ndarray]:
        all_metas = await self.store.list_skills()
        if not use_gnn:
            return {m.id: np.array(m.embedding) for m in all_metas if m.embedding}
        graph = self.store.get_graph()
        save_dir = self.store.workspace_path
        return encode_graph_embeddings(
            metas=all_metas, edges=graph.edges, embed_dim=384, save_dir=save_dir
        )

    async def update_skill_node(self, skill_id: str, meta: dict, store: BaseSkillStore):
        graph = self.store.get_graph()
        # Updates node data
        graph.nodes[skill_id] = meta
        await self.store.update_graph(graph)

async def register_skill_in_graph(skill_id, meta, all_metas, store, use_gnn: bool = False):
    graph = store.get_graph()

    def _val(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # Ensures we have an {id: meta} dictionary
    if isinstance(all_metas, list):
        metas_dict = {m.id if hasattr(m, 'id') else m.get('id'): m for m in all_metas}
    else:
        metas_dict = all_metas

    graph.nodes[skill_id] = {
        "title":    _val(meta, 'title'),
        "category": _val(meta, 'category'),
        "domain":   _val(meta, 'domain'),
    }

    category = _val(meta, 'category') or "General"
    # Now uses metas_dict
    for sid, other_meta in metas_dict.items():
        if sid == skill_id:
            continue
        other_cat = _val(other_meta, 'category') or "General"
        if category.split('/')[0] == other_cat.split('/')[0]:
            exists = any(
                (e['from'] == skill_id and e['to'] == sid) or
                (e['from'] == sid and e['to'] == skill_id)
                for e in graph.edges
            )
            if not exists:
                graph.edges.append({
                    "from": skill_id, "to": sid,
                    "type": "SIMILAR_TO", "weight": 0.7,
                    "reason": f"Category match: {category}",
                })

    await store.update_graph(graph)

    if use_gnn:
        log.info("graph.gnn_refinement_start", trigger_skill=skill_id)
        current_all_metas = await store.list_skills()
        save_dir = store.workspace_path
        updated_vectors = encode_graph_embeddings(
            metas=current_all_metas, edges=graph.edges, embed_dim=384, save_dir=save_dir
        )
        from openskill.core.vector import TurboQuantizer, pack_qvector
        _quantizer = TurboQuantizer(dimension=384)
        for sid, new_vec in updated_vectors.items():
            vec_norm = np.linalg.norm(new_vec)
            if vec_norm < 1e-6:
                log.warning("graph.gnn_zero_vector_skipped", skill_id=sid)
                continue
            qv = _quantizer.quantize(new_vec)
            await store.save_embedding(
                skill_id=sid, embedding=new_vec.tolist(),
                qvector=pack_qvector(qv), model_name="GNN_MiniLM",
                dimension=384, provider="OpenSkillGNN",
            )
        log.info("graph.gnn_refinement_done", total_nodes=len(updated_vectors))
