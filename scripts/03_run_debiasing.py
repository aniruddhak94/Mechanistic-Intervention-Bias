#!/usr/bin/env python3
"""
03_run_debiasing.py — Intervene on bias circuits and measure improvement.

Usage:
    python scripts/03_run_debiasing.py \\
        --model gpt2 \\
        --dataset data/gender_bias.json \\
        --edges results/top_edges_gender.json

Output:
    results/debiasing_results.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model
from src.data_utils import load_bias_dataset, create_prompt_pairs, get_gendered_token_ids
from src.baseline_scoring import run_baseline
from src.eap_algorithm import load_edges
from src.intervention import run_debiasing_comparison


def main():
    parser = argparse.ArgumentParser(description="Debias GPT-2 via edge ablation")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name: gpt2, gpt2-medium, gpt2-large")
    parser.add_argument("--dataset", type=str, default="data/gender_bias.json",
                        help="Path to bias dataset JSON")
    parser.add_argument("--edges", type=str, required=True,
                        help="Path to top edges JSON (from step 02)")
    parser.add_argument("--output", type=str, default="results/debiasing_results.json",
                        help="Output path for comparison results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 3: DEBIASING VIA EDGE ABLATION")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Edges:   {args.edges}")
    print(f"  Output:  {args.output}")
    print("=" * 60)

    # Load model
    model = load_model(args.model, device=args.device)

    # Load and tokenize dataset
    raw_data = load_bias_dataset(args.dataset)
    prompt_pairs = create_prompt_pairs(raw_data, model)

    # Get gendered tokens
    male_ids, female_ids = get_gendered_token_ids(model)

    # Load top edges
    edges = load_edges(args.edges)

    # First: compute baseline (or load if exists)
    baseline_path = args.output.replace("debiasing_results", "baseline_scores")
    baseline_path = baseline_path.replace("debiasing_", "baseline_")
    print("\n[Step 3a] Computing baseline bias scores...")
    baseline = run_baseline(model, prompt_pairs, male_ids, female_ids)

    # Then: run debiasing comparison
    print("\n[Step 3b] Running debiased forward passes...")
    comparison = run_debiasing_comparison(
        model=model,
        edges=edges,
        prompt_pairs=prompt_pairs,
        male_ids=male_ids,
        female_ids=female_ids,
        baseline_results=baseline,
        save_path=args.output,
    )

    # Detailed per-prompt output
    print(f"\n{'─' * 70}")
    print(f"  {'Prompt':<40s}  {'Before':>8s}  {'After':>8s}  {'Δ':>8s}")
    print(f"{'─' * 70}")
    for item in comparison["per_prompt"]:
        prompt_short = item["clean_prompt"][:38] + ".." if len(item["clean_prompt"]) > 40 else item["clean_prompt"]
        delta = item["reduction"]
        symbol = "↓" if delta > 0 else "↑"
        print(f"  {prompt_short:<40s}  {item['bias_before']:8.4f}  {item['bias_after']:8.4f}  {symbol}{abs(delta):7.4f}")

    print(f"\n✓ Debiasing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
