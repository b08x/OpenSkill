"""
train_scorer.py — Complete Path Scorer Training Script
===========================================================
Replaces the `openskill train-bootstrap` command with a more
robust pipeline: generates data → trains → evaluates → saves.

FIX: Auto-detects skill embedding dimension.
     No longer requires manual --embed-dim.

Usage:
    # Full training (auto-detected dimension)
    python train_scorer.py --skill-dir ./skills_output

    # Reuse already generated dataset (faster iterations)
    python train_scorer.py --skill-dir ./skills_output --data-cache train_data.npz

    # Just generate dataset without training
    python train_scorer.py --skill-dir ./skills_output --only-data

    # Force specific dimension (if auto-detection fails)
    python train_scorer.py --skill-dir ./skills_output --embed-dim 1536

    # Customize epochs and learning rate
    python train_scorer.py --skill-dir ./skills_output --epochs 100 --lr 0.0003

Output:
    ./skills_output/path_scorer.safetensors   ← scorer ready for use
    ./train_data.npz                          ← cached dataset

The scorer is automatically loaded by SkillGraph on the next
call to find_paths() — no code changes needed.
"""

from __future__ import annotations

import asyncio
import argparse
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()


def _detect_dim_from_skills(skill_dir: str) -> int:
    """
    Detects embedding dimension by inspecting skill meta.json files.
    Returns the most common dimension (e.g., 1536 for OpenAI, 384 for local MiniLM).
    """
    import json
    skills_path = Path(skill_dir) / "skills"
    dim_counts: dict[int, int] = {}

    for meta_file in skills_path.rglob("meta.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            vectors = data.get("vectors", {})
            for profile in vectors.values():
                d = profile.get("dimension", 0)
                emb = profile.get("embedding")
                prov = profile.get("provider", "")
                if d > 0 and emb and prov != "OpenSkillGNN":
                    dim_counts[d] = dim_counts.get(d, 0) + 1
            # Fallback: root embedding field
            root_emb = data.get("embedding")
            if root_emb and isinstance(root_emb, list) and len(root_emb) > 0:
                d = len(root_emb)
                dim_counts[d] = dim_counts.get(d, 0) + 1
        except Exception:
            continue

    if not dim_counts:
        return 0  # signals "not found"

    detected = max(dim_counts, key=lambda d: dim_counts[d])
    print(f"      Dimensions found: {dict(sorted(dim_counts.items()))}")
    print(f"      Using: {detected}d ({dim_counts[detected]} skills)")
    return detected


async def run_training(args):
    """Full pipeline: data → train → evaluation."""

    print("\n" + "=" * 60)
    print("  OpenSkill Path Scorer — Bootstrap Training")
    print("=" * 60)

    # ── 0. Detect dimension ───────────────────────────────────────

    embed_dim = args.embed_dim
    if embed_dim == 0:
        print(f"\n[0/3] Auto-detecting embedding dimension in '{args.skill_dir}'...")
        embed_dim = _detect_dim_from_skills(args.skill_dir)
        if embed_dim == 0:
            print("\n  ERROR: No embeddings found in skills.")
            print("  Skills must have embeddings for scorer training.")
            print("\n  Run one of the commands below to generate embeddings:")
            print("    openskill embed <skill-id> --api-key <your-key>   (OpenAI 1536d)")
            print("    openskill embed <skill-id> --local                 (MiniLM 384d)")
            sys.exit(1)
        print(f"      ✓ Dimension detected: {embed_dim}d")
    else:
        print(f"\n[0/3] Using specified dimension: {embed_dim}d")

    # ── 1. Dataset ────────────────────────────────────────────────

    cache = Path(args.data_cache)

    if cache.exists() and not args.regen_data:
        print(f"\n[1/3] Loading dataset from cache: {cache}")
        from bootstrap_data import load_dataset
        train_data, val_data = load_dataset(str(cache))
        # Verify cache dimension matches detected
        if train_data:
            cached_dim = train_data[0][0].shape[0]
            if cached_dim != embed_dim:
                print(f"\n  WARNING: Cache has dim={cached_dim}d but skills have dim={embed_dim}d.")
                print("  Regenerating dataset...")
                cache.unlink()
                train_data, val_data = None, None

    if not cache.exists() or args.regen_data:
        print(f"\n[1/3] Generating synthetic dataset from '{args.skill_dir}'...")
        print("      (this may take 1-2 min to download the embedder for the first time)")
        from bootstrap_data import generate_bootstrap_dataset, save_dataset
        train_data, val_data = await generate_bootstrap_dataset(
            skill_dir=args.skill_dir,
            embed_dim=embed_dim,
            val_split=0.15,
            hard_neg_ratio=0.4,
            seed=args.seed,
        )
        save_dataset(train_data, val_data, str(cache))
        print(f"      Dataset saved to: {cache}")

    print(f"\n      Train: {len(train_data)} samples")
    print(f"      Val:   {len(val_data)} samples")
    pos = sum(1 for s in train_data if s[2])
    print(f"      Positives: {pos} ({pos/max(len(train_data),1)*100:.1f}%)")
    print(f"      Negatives: {len(train_data)-pos} ({(len(train_data)-pos)/max(len(train_data),1)*100:.1f}%)")
    print(f"      Dimension:  {embed_dim}d")

    if args.only_data:
        print("\n--only-data mode: dataset generated, skipping training.")
        return

    # ── 2. Training ─────────────────────────────────────────────────

    save_path = str(Path(args.skill_dir) / "path_scorer.safetensors")

    print(f"\n[2/3] Training PathScorerModel for {args.epochs} epochs...")
    print(f"      LR: {args.lr}  |  Batch: 32  |  InfoNCE + BCE verifier")
    print(f"      Gumbel temperature: 1.0 → 0.1 (annealed)")
    print(f"      Embed dim: {embed_dim}d")
    print()

    from openskill.core.trainer import train_path_scorer, evaluate_scorer

    metrics = await train_path_scorer(
        train_data=train_data,
        embed_dim=embed_dim,
        save_path=save_path,
        epochs=args.epochs,
        lr=args.lr,
    )

    print(f"\n      Best loss: {metrics['best_loss']:.4f}")
    print(f"      Model saved to: {save_path}")

    # ── 3. Evaluation ──────────────────────────────────────────────

    print(f"\n[3/3] Evaluating on validation set ({len(val_data)} samples)...")

    if len(val_data) >= 4:
        eval_metrics = evaluate_scorer(save_path, val_data, embed_dim=embed_dim)
        print(f"\n      Verifier Accuracy: {eval_metrics['verifier_accuracy']:.2%}")
        print(f"      MRR:               {eval_metrics['mrr']:.4f}")
        print(f"      Tested samples: {eval_metrics['n_test']}")

        if eval_metrics["verifier_accuracy"] < 0.6:
            print("\n  WARNING: accuracy below 60%. Consider:")
            print("         - More diverse skills (recommended: ≥5 different domains)")
            print("         - More epochs: --epochs 150")
            print("         - Check if embeddings were generated correctly")
    else:
        print("      (not enough validation data — evaluation skipped)")

    # ── Final Summary ─────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETED")
    print("=" * 60)
    print(f"\n  Scorer saved to: {save_path}")
    print(f"  Dimension: {embed_dim}d")
    print("\n  Next steps:")
    print("    1. Run a query: openskill retrieve --query 'your query'")
    print("    2. SkillGraph will load the scorer automatically")
    print("    3. Watch for 'graph.neural_scorer_loaded' in logs")
    print("    4. Confidence should rise from 0.00 to > 0.35")
    print("\n  To retrain with more data:")
    print(f"    python train_scorer.py --skill-dir {args.skill_dir} --regen-data --epochs {args.epochs}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Trains PathScorerModel (S-Path-RAG) from existing skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skill-dir",
        default="./skills_output",
        help="Skills folder (default: ./skills_output)",
    )
    parser.add_argument(
        "--data-cache",
        default="train_data.npz",
        help=".npz file for dataset cache (default: train_data.npz)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=0,
        help="Embedding dimension (0=auto-detect, default: 0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs (default: 80)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--regen-data",
        action="store_true",
        help="Regenerates dataset even if cache exists",
    )
    parser.add_argument(
        "--only-data",
        action="store_true",
        help="Only generates dataset, without training",
    )

    args = parser.parse_args()

    if not Path(args.skill_dir).exists():
        print(f"\nERROR: Folder '{args.skill_dir}' not found.")
        print("Create skills first with: openskill create --task 'your task'")
        sys.exit(1)

    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
