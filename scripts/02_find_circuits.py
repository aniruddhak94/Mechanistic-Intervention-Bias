#!/usr/bin/env python3
"""
02_find_circuits.py — Run Edge Attribution Patching to find bias circuits.

Usage:
    python scripts/02_find_circuits.py --model gpt2 --dataset data/gender_bias.json --top_k 50

Output:
    results/top_edges_gender.json  (or name derived from dataset)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model
from src.data_utils import load_bias_dataset, create_prompt_pairs, get_gendered_token_ids
from src.eap_algorithm import aggregate_eap_scores, get_top_edges, save_edges


def main():
    parser = argparse.ArgumentParser(description="Find bias circuits via EAP")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name: gpt2, gpt2-medium, gpt2-large")
    parser.add_argument("--dataset", type=str, default="data/gender_bias.json",
                        help="Path to bias dataset JSON")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top edges to keep")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for top edges JSON")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--min_layer", type=int, default=1,
                        help="Min source layer for EAP (default 1 to exclude L0)")
    args = parser.parse_args()

    # Auto-generate output path
    if args.output is None:
        ds_name = os.path.splitext(os.path.basename(args.dataset))[0]
        # e.g., "gender_bias" -> "top_edges_gender"
        short_name = ds_name.replace("_bias", "")
        args.output = f"results/top_edges_{short_name}.json"

    print("=" * 60)
    print("  STEP 2: EDGE ATTRIBUTION PATCHING (CIRCUIT DISCOVERY)")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Top-K:   {args.top_k}")
    print(f"  Output:  {args.output}")
    print(f"  Min Layer: {args.min_layer}")
    print("=" * 60)

    # Load model
    model = load_model(args.model, device=args.device)

    # Load and tokenize dataset
    raw_data = load_bias_dataset(args.dataset)
    prompt_pairs = create_prompt_pairs(raw_data, model)

    # Get gendered tokens for bias metric
    male_ids, female_ids = get_gendered_token_ids(model)

    # Run EAP across all prompt pairs
    print("\n[EAP] Starting Edge Attribution Patching...")
    print(f"[EAP] This will process {len(prompt_pairs)} prompt pairs.")
    print(f"[EAP] Estimated time: {len(prompt_pairs) * 10}-{len(prompt_pairs) * 30}s on GPU\n")

    all_edges = aggregate_eap_scores(model, prompt_pairs, male_ids, female_ids, min_layer=args.min_layer)

    # Extract top-K edges
    top_edges = get_top_edges(all_edges, args.top_k)

    # Save
    save_edges(top_edges, args.output)

    # Print summary
    print(f"\n{'-' * 60}")
    print(f"  Top {args.top_k} Bias-Causing Edges:")
    print(f"{'-' * 60}")
    for i, edge in enumerate(top_edges[:20]):
        print(f"  {i + 1:3d}. {edge}")
    if len(top_edges) > 20:
        print(f"  ... ({len(top_edges) - 20} more edges in {args.output})")

    # Layer distribution summary
    src_layer_counts = {}
    for e in top_edges:
        src_layer_counts[e.src_layer] = src_layer_counts.get(e.src_layer, 0) + 1
    print(f"\n  Source layer distribution:")
    for layer in sorted(src_layer_counts.keys()):
        bar = "#" * src_layer_counts[layer]
        print(f"    Layer {layer:2d}: {bar} ({src_layer_counts[layer]})")

    print(f"\n[OK] Circuit discovery complete. Top edges saved to {args.output}")


if __name__ == "__main__":
    main()
