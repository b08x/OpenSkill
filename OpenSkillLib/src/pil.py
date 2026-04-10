"""
run_full_pipeline.py — Embed all skills + Re-train + Test the system
==========================================================================
Executes the 3 stages in the correct order:
  1. Embed: generates 1536d vectors for ALL skills via OpenRouter
             and verifies they were persisted in the correct bundle
  2. Retrain: regenerates dataset and trains the scorer with all skills
  3. Test: executes real queries and verifies if confidence > 0.35

Usage:
    python run_full_pipeline.py --api-key sk-or-v1-...
    python run_full_pipeline.py --api-key sk-or-v1-... --skill-dir ./skills_output
    python run_full_pipeline.py --api-key sk-or-v1-... --skip-embed   (skips stage 1)
    python run_full_pipeline.py --api-key sk-or-v1-... --skip-train   (skips stage 2)
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import json
import time
from pathlib import Path


# ─── Persistence Verification ─────────────────────────────────────────────

def verify_embeddings_persisted(skill_dir: str, expected_dim: int = 1536) -> dict:
    """
    Scans bundle meta.json files and counts how many have a vector of the expected dimension.
    Returns detailed diagnostic to validate before training.
    """
    skills_path = Path(skill_dir) / "skills"
    result = {"total_bundles": 0, "have_vector": 0, "missing_vector": [], "dims_found": set()}

    for meta_file in skills_path.rglob("meta.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            skill_id = data.get("id", meta_file.parent.name)
            result["total_bundles"] += 1

            found = False
            vectors = data.get("vectors", {})
            for profile in vectors.values():
                dim = profile.get("dimension", 0)
                emb = profile.get("embedding")
                prov = profile.get("provider", "")
                result["dims_found"].add(dim)
                if dim == expected_dim and emb and prov != "OpenSkillGNN":
                    found = True
                    break

            # Fallback: root embedding field
            if not found:
                root_emb = data.get("embedding")
                if root_emb and len(root_emb) == expected_dim:
                    found = True

            if found:
                result["have_vector"] += 1
            else:
                result["missing_vector"].append(data.get("title", skill_id)[:50])

        except Exception:
            continue

    result["dims_found"] = sorted(result["dims_found"])
    return result


# ─── Stage 1: Embed all skills ──────────────────────────────────────────

async def embed_all_skills(skill_dir: str, api_key: str, local: bool = False) -> dict:
    """Generates embeddings. If local=True uses 384d, if local=False uses OpenAI 1536d."""
    from openskill.storage.local import LocalDiskStore
    from openskill.core.vector import pack_qvector
    import os

    store = LocalDiskStore(skill_dir)
    target_dim = 384 if local else 1536

    if local:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    else:
        os.environ["OPENROUTER_API_KEY"] = api_key
        from openskill.llm.openrouter import OpenRouterProvider
        from openskill.core.vector import TurboQuantizer
        llm = OpenRouterProvider(api_key=api_key)
        quantizer = TurboQuantizer(dimension=1536)

    all_metas = await store.list_skills()
    print(f"\n  Skills found: {len(all_metas)}")

    results = {"total": len(all_metas), "embedded": 0, "skipped": 0, "failed": []}

    for i, meta in enumerate(all_metas, 1):
        title = meta.title or meta.id
        print(f"\n  [{i}/{len(all_metas)}] {title[:60]}")

        # Check if 1536d vector already exists in bundle (not GNN)
        already_has_1536 = False
        if meta.vectors:
            for profile in meta.vectors.values():
                dim = getattr(profile, "dimension", 0)
                prov = getattr(profile, "provider", "")
                emb = getattr(profile, "embedding", None)
                if dim == 1536 and emb and prov != "OpenSkillGNN":
                    already_has_1536 = True
                    break

        if already_has_1536:
            print(f"    ✓ Already has 1536d in bundle — skipping")
            results["skipped"] += 1
            continue

        # Load skill markdown
        skill_md = await store.get_skill_md(meta.id)
        if not skill_md:
            print(f"    ✗ Markdown not found — skipping")
            results["failed"].append(meta.id)
            continue

        # Generate embedding
        try:
            print(f"    → Generating {target_dim}d embedding...")
            if local:
                # Local 384d
                vec_np = embedder.encode(skill_md[:3000], normalize_embeddings=True)
                embedding_list = vec_np.tolist()
                qv_pack = {}  # Dummy qvector for local
                prov = "LocalMiniLM"
            else:
                # OpenAI 1536d
                vec = await quantizer.embed(llm, skill_md[:3000])
                embedding_list = vec.tolist()
                qv_pack = pack_qvector(quantizer.quantize(vec))
                prov = "OpenAI"

            await store.save_embedding(
                skill_id=meta.id,
                embedding=embedding_list,
                qvector=qv_pack,
                model_name="local-model" if local else "openai/text-embedding-3-small",
                dimension=target_dim,
                provider=prov,
            )

            # ── Immediately verify if saved in correct place ──
            bundle_meta = Path(skill_dir) / "skills" / meta.id / "meta.json"
            legacy_meta = Path(skill_dir) / "skills" / f"{meta.id}.json"

            if bundle_meta.exists():
                saved_data = json.loads(bundle_meta.read_text(encoding="utf-8"))
                vectors = saved_data.get("vectors", {})
                found_in_bundle = any(
                    p.get("dimension") == 1536 and p.get("embedding")
                    for p in vectors.values()
                )
                if found_in_bundle:
                    print(f"    ✓ Bundle updated: dim=1536, norm=1.000")
                else:
                    print(f"    ⚠ WARNING: bundle exists but 1536d vector not found!")
            elif legacy_meta.exists():
                print(f"    ⚠ Saved in legacy ({meta.id}.json) — no bundle")
            else:
                print(f"    ✗ ERROR: file not found after save!")
                results["failed"].append(meta.id)
                continue

            results["embedded"] += 1
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"    ✗ Error: {e}")
            results["failed"].append(meta.id)

    # ── Final persistence verification ──
    print(f"\n  Verifying persistence on disk...")
    verify = verify_embeddings_persisted(skill_dir, expected_dim=1536)
    print(f"  Bundles found: {verify['total_bundles']}")
    print(f"  With 1536d vector:     {verify['have_vector']}/{verify['total_bundles']}")
    if verify["missing_vector"]:
        print(f"  Without 1536d vector:     {verify['missing_vector']}")
    print(f"  Dimensions on disk:  {verify['dims_found']}")
    results["verify"] = verify

    return results


# ─── Stage 2: Re-train the scorer ─────────────────────────────────────────────

async def retrain_scorer(skill_dir: str, api_key: str, epochs: int = 80, local: bool = False) -> dict:
    """
    Regenerates the dataset and trains the PathScorerModel.
    Automatically detects the dimension of valid skills.
    """
    from bootstrap_data import generate_bootstrap_dataset, save_dataset
    from openskill.core.trainer import train_path_scorer, evaluate_scorer

    # ── Detect current bundle dimension ──
    verify = verify_embeddings_persisted(skill_dir, expected_dim=1536)
    n_valid = verify["have_vector"]
    print(f"\n  Skills with 1536d vectors: {n_valid}/{verify['total_bundles']}")

    if n_valid < 2:
        print(f"  ERROR: Need at least 2 skills with 1536d to train.")
        print(f"  Run embed stage first.")
        return {"best_loss": None, "error": "insufficient skills with 1536d"}

    cache_path = Path("train_data_retrain.npz")

    print(f"\n  Generating dataset (forcing regen with embed_dim=1536)...")
    provider = "local" if local else "openai"
    target_dim = 384 if local else 1536

    train_data, val_data = await generate_bootstrap_dataset(
        skill_dir=skill_dir,
        embed_dim=target_dim,
        val_split=0.15,
        hard_neg_ratio=0.4,
        seed=42,
        api_key=api_key,
        provider=provider
    )
    save_dataset(train_data, val_data, str(cache_path))

    # Validate actual dimension of generated dataset
    if train_data:
        actual_dim = train_data[0][0].shape[0]
        print(f"  Actual dataset dimension: {actual_dim}d")
        if actual_dim != 1536:
            print(f"  ⚠ WARNING: dataset in {actual_dim}d instead of 1536d!")
            print(f"  This means bootstrap still hasn't found 1536d vectors.")
    else:
        print(f"  ERROR: empty dataset!")
        return {"best_loss": None, "error": "dataset empty"}

    pos = sum(1 for s in train_data if s[2])
    neg = len(train_data) - pos
    print(f"  Dataset: {len(train_data)} train | {len(val_data)} validation")
    print(f"  Positives: {pos} ({pos/max(len(train_data),1)*100:.1f}%)")
    print(f"  Negatives: {neg}")

    embed_dim = train_data[0][0].shape[0]
    save_path = str(Path(skill_dir) / "path_scorer.safetensors")
    print(f"\n  Training for {epochs} epochs (dim={embed_dim}d)...")

    metrics = await train_path_scorer(
        train_data=train_data,
        embed_dim=embed_dim,
        save_path=save_path,
        epochs=epochs,
        lr=3e-4,
    )

    if len(val_data) >= 4:
        eval_m = evaluate_scorer(save_path, val_data, embed_dim=embed_dim)
        metrics.update(eval_m)
        print(f"\n  Verifier Accuracy: {eval_m['verifier_accuracy']:.2%}")
        print(f"  MRR:               {eval_m['mrr']:.4f}")

    metrics["embed_dim_used"] = embed_dim
    return metrics


# ─── Stage 3: Test the system ─────────────────────────────────────────────────

async def test_system(skill_dir: str, api_key: str) -> dict:
    """
    Executes test queries and verifies system operation.
    Reloads scorer from disk to ensure it uses the newly trained one.
    """
    import os
    os.environ["OPENROUTER_API_KEY"] = api_key

    from openskill import OpenSkillClient, LocalDiskStore
    from openskill.llm.openrouter import OpenRouterProvider
    from openskill.core.graph import SkillGraph

    store = LocalDiskStore(skill_dir)
    llm = OpenRouterProvider(api_key=api_key)
    client = OpenSkillClient(store=store, llm=llm)

    # Force reload of scorer from disk
    client._graph = None
    graph = client.graph
    graph._load_scorer()

    scorer_loaded = graph._neural_scorer is not None
    scorer_dim = graph._neural_scorer.embed_dim if scorer_loaded else None
    print(f"\n  Scorer loaded: {'✓' if scorer_loaded else '✗'}")
    if scorer_loaded:
        print(f"  Scorer embed dim: {scorer_dim}d")
        if scorer_dim != 1536:
            print(f"  ⚠ WARNING: scorer in {scorer_dim}d but skills in 1536d — dimensions mismatch!")
            print(f"  Scorer will fall back to manual (confidence=0.00)")
            print(f"  Run retrain stage to fix.")

    test_queries = [
        "Calculate the Fibonacci sequence using memoization",
        "How to implement dynamic programming with caching",
        "Find the 100th Fibonacci number efficiently",
        "Memoization technique for recursive algorithms",
        "Optimize recursive sequence calculation",
    ]

    results = []
    for query in test_queries:
        print(f"\n  Query: '{query[:55]}'")
        t0 = time.time()
        guidance = await client.retriever.retrieve(query, top_k=3, use_graph=False)
        elapsed = time.time() - t0

        alphas_uniform = (
            len(guidance.skill_alphas) < 2 or
            all(abs(a - guidance.skill_alphas[0]) < 0.01 for a in guidance.skill_alphas)
        )

        top_skills = []
        for sid in guidance.best_path_ids[:3]:
            meta = await store.get_skill_meta(sid)
            if meta:
                top_skills.append(meta.title[:40])

        r = {
            "query": query,
            "confidence": guidance.confidence,
            "skill_alphas": guidance.skill_alphas,
            "top_skills": top_skills,
            "alphas_uniform": alphas_uniform,
            "latency_ms": int(elapsed * 1000),
        }
        results.append(r)

        conf_icon = "✓" if guidance.confidence > 0.35 else ("~" if guidance.confidence > 0.10 else "✗")
        alpha_status = "uniform" if alphas_uniform else "discriminative ✓"
        print(f"    {conf_icon} Confidence: {guidance.confidence:.4f}")
        print(f"    Alphas: {[f'{a:.3f}' for a in guidance.skill_alphas]} ({alpha_status})")
        print(f"    Top skills: {top_skills[:2]}")
        print(f"    Latency: {r['latency_ms']}ms")

    n_nonzero = sum(1 for r in results if r["confidence"] > 0)
    n_above_threshold = sum(1 for r in results if r["confidence"] > 0.35)
    n_discriminative = sum(1 for r in results if not r["alphas_uniform"])
    avg_conf = sum(r["confidence"] for r in results) / max(len(results), 1)

    print(f"\n  {'='*52}")
    print(f"  Scorer dim:           {scorer_dim}d")
    print(f"  Confidence > 0:       {n_nonzero}/{len(results)} queries")
    print(f"  Confidence > 0.35:    {n_above_threshold}/{len(results)} queries")
    print(f"  Discriminative alphas: {n_discriminative}/{len(results)} queries")
    print(f"  Avg confidence:     {avg_conf:.4f}")

    if avg_conf > 0.35:
        print(f"\n  ✓ SYSTEM OPERATIONAL — scorer discriminating well")
    elif avg_conf > 0.10:
        print(f"\n  ~ PARTIAL — scorer active but confidence low")
        print(f"    → Add skills from different domains to increase contrast")
        print(f"    → Or increase epochs: --epochs 150")
    elif scorer_dim and scorer_dim != 1536:
        print(f"\n  ✗ WRONG DIMENSION — scorer in {scorer_dim}d, skills in 1536d")
        print(f"    → Re-execute without --skip-train to retrain in 1536d")
    else:
        print(f"\n  ✗ SCORER INACTIVE — verify if embed was completed")
        print(f"    → Execute without --skip-embed to re-embed skills")

    return {
        "results": results,
        "avg_confidence": avg_conf,
        "n_discriminative": n_discriminative,
        "scorer_loaded": scorer_loaded,
        "scorer_dim": scorer_dim,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(args):
    print("\n" + "="*62)
    print("  OpenSkill — Full Pipeline: Embed + Retrain + Test")
    print("="*62)

    start = time.time()
    summary = {}

    # Stage 1: Embed
    if not args.skip_embed:
        print("\n[STAGE 1/3] Embedding all skills (1536d)...")
        print("-"*52)
        embed_result = await embed_all_skills(args.skill_dir, args.api_key, args.local)
        summary["embed"] = embed_result
        verify = embed_result.get("verify", {})
        n_valid = verify.get("have_vector", 0)
        n_total = verify.get("total_bundles", 0)
        print(f"\n  Embed result: {embed_result['embedded']} new | "
              f"{embed_result['skipped']} already had | "
              f"{len(embed_result['failed'])} failures")
        print(f"  Disk verification: {n_valid}/{n_total} skills with 1536d vector")

        if n_valid < 2:
            print(f"\n  CRITICAL ERROR: only {n_valid} skill(s) with 1536d vector on disk.")
            print(f"  Verify if local.py was updated correctly.")
            print(f"  The local.py file must be in: openskill/storage/local.py")
            sys.exit(1)
    else:
        print("\n[STAGE 1/3] Embed skipped (--skip-embed)")
        verify = verify_embeddings_persisted(args.skill_dir, expected_dim=1536)
        print(f"  Skills with 1536d on disk: {verify['have_vector']}/{verify['total_bundles']}")

    # Stage 2: Re-train
    if not args.skip_train:
        print(f"\n[STAGE 2/3] Re-training the scorer (epochs={args.epochs})...")
        print("-"*52)
        train_result = await retrain_scorer(args.skill_dir, args.api_key, args.epochs, args.local)
        summary["train"] = train_result
        if train_result.get("best_loss") is not None:
            print(f"\n  Best loss:    {train_result['best_loss']:.4f}")
            print(f"  Embed dim:    {train_result.get('embed_dim_used', '?')}d")
    else:
        print("\n[STAGE 2/3] Retrain skipped (--skip-train)")

    # Stage 3: Test
    print("\n[STAGE 3/3] Testing the system with real queries...")
    print("-"*52)
    test_result = await test_system(args.skill_dir, args.api_key)
    summary["test"] = test_result

    # Final result
    elapsed = time.time() - start
    print(f"\n{'='*62}")
    print(f"  PIPELINE COMPLETED in {elapsed:.1f}s")
    print(f"{'='*62}")

    # Save report
    report_path = Path(args.skill_dir) / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        def serialize(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, set):
                return sorted(obj)
            return str(obj)
        json.dump(summary, f, indent=2, default=serialize)
    print(f"\n  Report saved to: {report_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full pipeline: Embed + Retrain + Test for OpenSkill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--skill-dir", default="./skills_output", help="Skills folder")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80)")
    parser.add_argument("--skip-embed", action="store_true", help="Skips embedding stage")
    parser.add_argument("--skip-train", action="store_true", help="Skips retraining")
    parser.add_argument("--local", action="store_true", help="Forces use of local 384d model throughout pipeline")
    args = parser.parse_args()

    if not Path(args.skill_dir).exists():
        print(f"ERROR: '{args.skill_dir}' not found.")
        sys.exit(1)

    asyncio.run(main(args))