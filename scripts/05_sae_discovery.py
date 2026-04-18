#!/usr/bin/env python3
"""
05_sae_discovery.py -- SAE gender feature discovery.

Loads pre-trained Sparse Autoencoders for target layers of GPT-2,
identifies gender-associated features, and saves them for intervention.

Usage (Kaggle):
    python scripts/05_sae_discovery.py --model gpt2 --dataset data/gender_bias.json

Output:
    results/gender_features.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model
from src.data_utils import load_bias_dataset, create_prompt_pairs, get_gendered_token_ids
from src.sae_analysis import (
    discover_gender_features_multilayer,
    compute_feature_statistics,
)


def main():
    parser = argparse.ArgumentParser(description="SAE gender feature discovery")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="data/gender_bias.json")
    parser.add_argument("--layers", type=str, default="6,8,9,10,11",
                        help="Comma-separated layers to analyze")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top features per layer")
    parser.add_argument("--output", type=str, default="results/gender_features.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print("=" * 60)
    print("  STEP 5: SAE GENDER FEATURE DISCOVERY (V3)")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Layers:  {layers}")
    print(f"  Top-K:   {args.top_k}")
    print(f"  Output:  {args.output}")
    print("=" * 60)

    model = load_model(args.model, device=args.device)
    bias_data = load_bias_dataset(args.dataset)
    prompt_pairs = create_prompt_pairs(bias_data, model)
    male_ids, female_ids = get_gendered_token_ids(model)

    # Run SAE feature discovery
    gender_features = discover_gender_features_multilayer(
        model, prompt_pairs, male_ids, female_ids,
        layers=layers, top_k_per_layer=args.top_k,
    )

    # Compute statistics
    stats = compute_feature_statistics(gender_features)
    gender_features["statistics"] = stats

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(gender_features, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  SAE DISCOVERY RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total gender features: {stats['total_gender_features_found']}")
    print(f"  Gender feature ratio:  {stats['gender_feature_ratio']:.4f}%")
    print(f"  Mean gender score:     {stats['mean_gender_score']:.6f}")
    print(f"  Male-biased features:  {stats['male_biased_count']}")
    print(f"  Female-biased features:{stats['female_biased_count']}")
    print(f"{'=' * 60}")
    print(f"\n[OK] Gender features saved to {args.output}")


if __name__ == "__main__":
    main()
