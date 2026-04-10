#!/usr/bin/env python3
"""
01_run_baseline.py — Measure baseline bias scores for GPT-2.

Usage:
    python scripts/01_run_baseline.py --model gpt2 --dataset data/gender_bias.json

Output:
    results/baseline_scores.json
"""

import argparse
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model
from src.data_utils import load_bias_dataset, create_prompt_pairs, get_gendered_token_ids
from src.baseline_scoring import run_baseline


def main():
    parser = argparse.ArgumentParser(description="Measure baseline bias in GPT-2")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name: gpt2, gpt2-medium, gpt2-large")
    parser.add_argument("--dataset", type=str, default="data/gender_bias.json",
                        help="Path to bias dataset JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results (default: auto-generated)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    # Auto-generate output path from dataset name
    if args.output is None:
        ds_name = os.path.splitext(os.path.basename(args.dataset))[0]
        args.output = f"results/baseline_{ds_name}.json"

    print("=" * 60)
    print("  STEP 1: BASELINE BIAS MEASUREMENT")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output:  {args.output}")
    print("=" * 60)

    # Load model
    model = load_model(args.model, device=args.device)

    # Load dataset
    raw_data = load_bias_dataset(args.dataset)

    # Create tokenized prompt pairs
    prompt_pairs = create_prompt_pairs(raw_data, model)

    # Get gendered token IDs
    male_ids, female_ids = get_gendered_token_ids(model)

    # Compute baseline bias
    results = run_baseline(
        model=model,
        dataset=prompt_pairs,
        male_ids=male_ids,
        female_ids=female_ids,
        save_path=args.output,
    )

    # Print per-prompt summary
    print(f"\n{'-' * 60}")
    print(f"  Per-Prompt Bias Scores:")
    print(f"{'-' * 60}")
    for r in results["per_prompt"]:
        direction_symbol = "M" if r["direction"] == "male" else "F"
        print(f"  {direction_symbol} {r['bias_score']:.4f}  |  {r['clean_prompt'][:50]}...")

    print(f"\n[OK] Baseline complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
