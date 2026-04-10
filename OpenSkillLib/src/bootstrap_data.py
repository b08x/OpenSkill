"""
bootstrap_data.py — Synthetic Dataset Generator for Path Scorer
====================================================================
Generates (query, path, label) pairs for training the PathScorerModel
without depending on external APIs. Uses sentence-transformers locally.

FIX: Auto-detects skill embedding dimension instead of
          assuming fixed 384d. Compatible with 384d (local MiniLM)
          and 1536d (OpenAI text-embedding-3-small via OpenRouter) embeddings.

Generation Strategy (3 types of negatives):
  1. Random negative   — path of a completely different skill
  2. Hard negative     — path of a skill in the same category (more confusing)
  3. Reversed negative — same positive path but wrong query (invariance)

Standalone Use:
    python bootstrap_data.py --skill-dir ./skills_output --output train_data.npz

Or Imported:
    from bootstrap_data import generate_bootstrap_dataset
    train_data, val_data = generate_bootstrap_dataset("./skills_output")
"""

from __future__ import annotations

import asyncio
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Any
import structlog

log = structlog.get_logger()

TrainSample = Tuple[np.ndarray, np.ndarray, bool]  # (query_vec, path_vec, is_positive)

# Synthetic queries per domain — broad coverage without needing LLM
DOMAIN_QUERIES: dict[str, list[str]] = {
    "Database": [
        "database connection is timing out and failing",
        "how to handle connection pool exhaustion",
        "SQL query running slow on large tables",
        "optimize index for faster reads",
        "deadlock detected between concurrent transactions",
        "how to batch insert millions of rows efficiently",
        "database is down, how to check health",
        "connection refused error from application",
        "how to run migrations without downtime",
        "query planner is not using the right index",
    ],
    "API": [
        "REST API returning 500 internal server error",
        "rate limiting users on my API endpoint",
        "how to version a REST API without breaking clients",
        "authentication failing with JWT token",
        "CORS error when calling API from browser",
        "API response is too slow, how to profile",
        "webhook not being received by my server",
        "how to document an API with OpenAPI spec",
        "handling large file uploads in API",
        "API gateway timeout configuration",
    ],
    "Python": [
        "memory leak in long running Python process",
        "asyncio event loop is blocking",
        "how to profile CPU usage in Python",
        "dependency conflict between packages",
        "how to write unit tests with pytest",
        "packaging a Python library for PyPI",
        "GIL limiting performance in multithreaded code",
        "how to use dataclasses effectively",
        "logging best practices in Python services",
        "debugging segfault in Python C extension",
    ],
    "Machine Learning": [
        "model accuracy is dropping in production",
        "training loss not converging after many epochs",
        "how to detect data drift in ML pipeline",
        "overfitting on training data despite dropout",
        "GPU out of memory during training",
        "how to evaluate a classification model",
        "feature importance for tree-based models",
        "how to serialize and deploy a PyTorch model",
        "batch normalization causing issues at inference",
        "how to handle class imbalance in training",
    ],
    "DevOps": [
        "container is crashing after deployment",
        "kubernetes pod in CrashLoopBackOff",
        "CI pipeline failing on test stage",
        "how to rollback a bad deployment",
        "memory limit exceeded in production pod",
        "how to set up blue-green deployment",
        "monitoring alerting on high error rate",
        "service discovery not working between pods",
        "secrets management in kubernetes",
        "how to drain a node safely",
    ],
    "Security": [
        "SQL injection vulnerability in login form",
        "how to implement OAuth2 correctly",
        "API key exposed in git history",
        "XSS attack prevention in web app",
        "how to rotate credentials without downtime",
        "insecure direct object reference vulnerability",
        "certificate expired in production",
        "how to audit user permissions",
        "brute force attack on login endpoint",
        "SSRF vulnerability in outbound requests",
    ],
    "Mathematics": [
        "how to calculate fibonacci sequence efficiently",
        "memoization technique for recursive algorithms",
        "dynamic programming approach to optimization",
        "how to implement recursive algorithms with cache",
        "sequence calculation with optimal time complexity",
        "storing intermediate results to avoid recomputation",
        "bottom-up vs top-down dynamic programming",
        "calculating nth term of a mathematical sequence",
        "optimal algorithm for sequence generation",
        "recursive function with memoization pattern",
    ],
    "Programming": [
        "how to implement memoization in Python",
        "caching results of expensive function calls",
        "recursive algorithm optimization techniques",
        "implementing dynamic programming solutions",
        "time complexity improvement with caching",
        "how to use lru_cache decorator",
        "space-time tradeoff in algorithms",
        "iterative vs recursive implementation",
        "algorithm performance optimization",
        "function result caching strategies",
    ],
    "General": [
        "how to debug this error",
        "what is the best approach for this problem",
        "step by step solution needed",
        "explain how this works",
        "what causes this issue",
        "how to fix this bug",
        "performance problem in production",
        "best practices for this scenario",
        "how to test this feature",
        "what tools should I use",
    ],
}

# Query templates using skill title directly
TITLE_QUERY_TEMPLATES = [
    "how to {title_lower}",
    "best way to {title_lower}",
    "{title_lower} step by step",
    "error when trying to {title_lower}",
    "guide for {title_lower}",
    "what is the correct approach to {title_lower}",
    "I need help with {title_lower}",
    "troubleshooting {title_lower}",
    "{title_lower} not working",
    "how do I {title_lower} correctly",
]


def _detect_embed_dim(all_metas) -> int:
    """
    Auto-detects embedding dimension from available skills.
    Prioritizes 1536d (OpenAI) over 384d (local MiniLM), as it's what
    OpenRouterProvider generates by default.
    """
    dim_counts: dict[int, int] = {}
    for m in all_metas:
        vectors = getattr(m, 'vectors', {}) or {}
        if isinstance(vectors, dict):
            for p in vectors.values():
                p_dim = getattr(p, 'dimension', 0) or (p.get('dimension', 0) if isinstance(p, dict) else 0)
                p_emb = getattr(p, 'embedding', None) or (p.get('embedding') if isinstance(p, dict) else None)
                p_prov = getattr(p, 'provider', '') or (p.get('provider', '') if isinstance(p, dict) else '')
                if p_dim > 0 and p_emb is not None and p_prov != "OpenSkillGNN":
                    dim_counts[p_dim] = dim_counts.get(p_dim, 0) + 1
        # Fallback: root embedding field
        root_emb = getattr(m, 'embedding', None)
        if root_emb and isinstance(root_emb, list) and len(root_emb) > 0:
            d = len(root_emb)
            dim_counts[d] = dim_counts.get(d, 0) + 1

    if not dim_counts:
        return 384  # default fallback

    # Return most common dimension
    detected = max(dim_counts, key=lambda d: dim_counts[d])
    log.info("bootstrap.detected_embed_dim", dim=detected, counts=dim_counts)
    return detected


async def _embed_queries_async(queries: list[str], target_dim: int, provider: str, api_key: str = None) -> np.ndarray:
    """
    Vectorizes synthetic queries using the correct provider to maintain the same latent space as skills.
    """
    # If OpenAI (1536d)
    if provider == "openai" or (provider == "auto" and target_dim == 1536):
        if not api_key:
            raise ValueError("API Key is required to vectorize queries with OpenAI.")

        from openskill.llm.openrouter import OpenRouterProvider
        import os
        os.environ["OPENROUTER_API_KEY"] = api_key
        llm = OpenRouterProvider(api_key=api_key)

        vecs = []
        log.info("bootstrap.embedding_openai", n_queries=len(queries))
        for q in queries:
            vec = await llm.embed(q)
            vecs.append(vec)
            await asyncio.sleep(0.05)  # Slight pause to avoid Rate Limit on API

        return np.array(vecs, dtype=np.float32)

    # If Local (384d)
    else:
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        log.info("bootstrap.embedding_local", model=model_name, n_queries=len(queries))
        model = SentenceTransformer(model_name)
        vecs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
        return vecs


def _queries_for_skill(meta: Any) -> list[str]:
    """Generates queries for a skill: combines templates with title + domain queries."""
    # Type checking for IDE
    if isinstance(meta, dict):
        title = meta.get("title", "") or ""
        category = meta.get("category", "General") or "General"
        task = meta.get("task", "") or ""
    else:
        title = getattr(meta, "title", "") or ""
        category = getattr(meta, "category", "General") or "General"
        task = getattr(meta, "task", "") or ""

    queries = []

    # 1. Title-based queries
    title_lower = title.lower().replace("_", " ").replace("-", " ")
    if title_lower:
        for tpl in TITLE_QUERY_TEMPLATES:
            queries.append(tpl.format(title_lower=title_lower))

    # 2. Domain queries by category
    domain_key = "General"
    for k in DOMAIN_QUERIES:
        if k.lower() in category.lower():
            domain_key = k
            break

    # Also try by title (e.g., "Fibonacci" → "Mathematics")
    if domain_key == "General":
        title_lower_check = title.lower()
        if any(w in title_lower_check for w in ["fibonacci", "sequence", "math", "calcul"]):
            domain_key = "Mathematics"
        elif any(w in title_lower_check for w in ["algorithm", "memoiz", "dynamic", "program"]):
            domain_key = "Programming"

    queries.extend(DOMAIN_QUERIES[domain_key])

    # 3. Fragments from task field (if available)
    if task:
        sentences = [s.strip() for s in task.replace("\n", ".").split(".") if len(s.strip()) > 20]
        queries.extend(sentences[:5])

    # Remove duplicates preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique


def _get_skill_vec(meta: Any, dim: int) -> Optional[np.ndarray]:
    """
    Extracts embedding of correct dimension from a SkillMetadata or dict.
    Ignores GNN vectors (provider=OpenSkillGNN).
    """
    # Safe extraction with explicit if/else
    if isinstance(meta, dict):
        vectors = meta.get('vectors', {}) or {}
        root_emb = meta.get('embedding', None)
    else:
        vectors = getattr(meta, 'vectors', {}) or {}
        root_emb = getattr(meta, 'embedding', None)

    if isinstance(vectors, dict):
        for p in vectors.values():
            if isinstance(p, dict):
                p_dim = p.get('dimension', 0) or 0
                p_emb = p.get('embedding')
                p_prov = p.get('provider', '') or ''
            else:
                p_dim = getattr(p, 'dimension', 0) or 0
                p_emb = getattr(p, 'embedding', None)
                p_prov = getattr(p, 'provider', '') or ''

            if p_dim == dim and p_emb is not None and p_prov != "OpenSkillGNN":
                return np.array(p_emb, dtype=np.float32)

    # Fallback: root embedding field
    if root_emb and isinstance(root_emb, list) and len(root_emb) == dim:
        return np.array(root_emb, dtype=np.float32)

    return None


async def generate_bootstrap_dataset(
    skill_dir: str = "./skills_output",
    embed_dim: int = 0,
    val_split: float = 0.15,
    hard_neg_ratio: float = 0.4,
    seed: int = 42,
    api_key: Optional[str] = None, # NEW PARAMETER
    provider: str = "auto"         # NEW PARAMETER
) -> Tuple[List[TrainSample], List[TrainSample]]:
    """
    Generates complete training and validation dataset.
    If embed_dim=0 (default), automatically detects skill dimension.

    Returns:
        (train_data, val_data) — lists of (query_vec, path_vec, is_positive)
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Load skills
    from openskill.storage.local import LocalDiskStore
    store = LocalDiskStore(skill_dir)
    all_metas = await store.list_skills()

    if len(all_metas) < 2:
        raise ValueError(
            f"Found only {len(all_metas)} skills in '{skill_dir}'.\n"
            "Contrastive training needs at least 2 skills.\n"
            "Create more skills with: openskill create"
        )

    log.info("bootstrap.skills_loaded", n=len(all_metas))

    # 2. Auto-detect dimension if not specified
    if embed_dim == 0:
        embed_dim = _detect_embed_dim(all_metas)
        log.info("bootstrap.auto_dim", embed_dim=embed_dim)

    # 3. Filter skills with correct dimension vector
    valid_metas: list[Tuple[Any, np.ndarray]] = []
    for m in all_metas:
        vec = _get_skill_vec(m, embed_dim)
        if vec is not None:
            valid_metas.append((m, vec))
        else:
            title = getattr(m, "title", "?")
            log.warning("bootstrap.skill_no_vector", title=title, expected_dim=embed_dim)

    if len(valid_metas) < 2:
        # Detailed diagnostic
        dims_found = set()
        for m in all_metas:
            vectors = getattr(m, 'vectors', {}) or {}
            for p in vectors.values():
                d = getattr(p, 'dimension', 0) or (p.get('dimension', 0) if isinstance(p, dict) else 0)
                if d > 0:
                    dims_found.add(d)
        raise ValueError(
            f"No skills have {embed_dim}d vector.\n"
            f"Dimensions found in skills: {dims_found or 'none'}\n\n"
            "Solutions:\n"
            f"  1. If your skills have {list(dims_found)[0] if dims_found else '?'}d dim, "
            f"pass --embed-dim {list(dims_found)[0] if dims_found else '1536'}\n"
            "  2. If there are no embeddings, run: openskill embed <skill-id> --api-key <key>\n"
            "  3. Use local mode: openskill embed <skill-id> --local"
        )

    log.info("bootstrap.valid_skills", n=len(valid_metas), embed_dim=embed_dim)

    # 4. Load embedder (with dimension auto-adjustment)


    # 5. Build category index → skills (for hard negatives)
    cat_index: dict[str, list[int]] = {}
    for i, (meta, _) in enumerate(valid_metas):
        cat = (getattr(meta, "category", "") or "General").split("/")[0]
        cat_index.setdefault(cat, []).append(i)

    # 6. Generate samples

    all_samples: List[TrainSample] = []

    for skill_idx, (meta, skill_vec) in enumerate(valid_metas):
        queries = _queries_for_skill(meta)
        log.info("bootstrap.generating", skill=getattr(meta, "title", "?"), n_queries=len(queries))

        # HERE IS THE MAGIC
        q_vecs = await _embed_queries_async(queries, embed_dim, provider, api_key)

        for q_vec in q_vecs:
            # Positive
            all_samples.append((q_vec, skill_vec, True))

            # Negatives
            other_indices = [i for i in range(len(valid_metas)) if i != skill_idx]
            if not other_indices:
                continue

            cat = (getattr(meta, "category", "") or "General").split("/")[0]
            same_cat = [i for i in cat_index.get(cat, []) if i != skill_idx]

            n_negs = 2
            n_hard = max(1, int(n_negs * hard_neg_ratio)) if same_cat else 0
            n_rand = n_negs - n_hard

            hard_neg_idx = random.sample(same_cat, min(n_hard, len(same_cat))) if same_cat else []
            rand_neg_idx_pool = [i for i in other_indices if i not in hard_neg_idx]
            rand_neg_idx = random.sample(rand_neg_idx_pool, min(n_rand, len(rand_neg_idx_pool)))

            for neg_i in hard_neg_idx + rand_neg_idx:
                all_samples.append((q_vec, valid_metas[neg_i][1], False))

            # Reversed negative (FIXED)
            wrong_query_skill_i = random.choice(other_indices)
            wrong_queries = _queries_for_skill(valid_metas[wrong_query_skill_i][0])
            if wrong_queries:
                wq = wrong_queries[random.randint(0, len(wrong_queries) - 1)]
                wq_vec = (await _embed_queries_async([wq], embed_dim, provider, api_key))[0]
                all_samples.append((wq_vec, skill_vec, False))

    log.info("bootstrap.total_samples", n=len(all_samples), embed_dim=embed_dim)

    # 7. Shuffle and split train/val
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * val_split))
    val_data = all_samples[:n_val]
    train_data = all_samples[n_val:]

    pos_train = sum(1 for s in train_data if s[2])
    neg_train = len(train_data) - pos_train
    log.info(
        "bootstrap.split",
        train=len(train_data),
        val=len(val_data),
        pos=pos_train,
        neg=neg_train,
        balance=f"{pos_train/max(len(train_data),1)*100:.1f}% positive",
    )

    return train_data, val_data


def save_dataset(
    train_data: List[TrainSample],
    val_data: List[TrainSample],
    path: str = "train_data.npz",
):
    """Saves dataset to .npz for reuse without re-generating."""
    def _pack(data):
        q = np.array([d[0] for d in data], dtype=np.float32)
        p = np.array([d[1] for d in data], dtype=np.float32)
        y = np.array([float(d[2]) for d in data], dtype=np.float32)
        return q, p, y

    tq, tp, ty = _pack(train_data)
    vq, vp, vy = _pack(val_data)
    np.savez(path, tq=tq, tp=tp, ty=ty, vq=vq, vp=vp, vy=vy)
    log.info("bootstrap.saved", path=path)


def load_dataset(path: str) -> Tuple[List[TrainSample], List[TrainSample]]:
    """Loads dataset saved in .npz."""
    d = np.load(path)
    train = list(zip(d["tq"], d["tp"], d["ty"].astype(bool)))
    val   = list(zip(d["vq"], d["vp"], d["vy"].astype(bool)))
    log.info("bootstrap.loaded", path=path, train=len(train), val=len(val))
    return train, val


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates bootstrap dataset for Path Scorer")
    parser.add_argument("--skill-dir", default="./skills_output", help="Skills folder")
    parser.add_argument("--output", default="train_data.npz", help="Output .npz file")
    parser.add_argument("--embed-dim", type=int, default=0,
                        help="Embedding dimension (0=auto-detect, default: 0)")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--hard-neg-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    async def _main():
        train, val = await generate_bootstrap_dataset(
            skill_dir=args.skill_dir,
            embed_dim=args.embed_dim,
            val_split=args.val_split,
            hard_neg_ratio=args.hard_neg_ratio,
            seed=args.seed,
        )
        save_dataset(train, val, args.output)
        dim = train[0][0].shape[0] if train else "?"
        print(f"\nDataset generated: {len(train)} train + {len(val)} validation")
        print(f"Vector dimension: {dim}d")
        print(f"Saved to: {args.output}")

    asyncio.run(_main())
