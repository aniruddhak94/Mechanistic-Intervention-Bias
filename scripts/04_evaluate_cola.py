#!/usr/bin/env python3
"""
04_evaluate_cola.py -- Check for collateral damage on CoLA.

V2: Uses MEAN ABLATION instead of zero ablation, and supports
alpha blending for partial intervention.

Usage:
    python scripts/04_evaluate_cola.py --model gpt2 --edges results/top_edges_gender.json --alpha 0.5

Output:
    results/cola_evaluation.json
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model
from src.data_utils import load_cola_dataset
from src.eap_algorithm import load_edges
from src.intervention import compute_mean_activations, build_mean_ablation_hooks


def evaluate_perplexity(model, sentences, max_samples=200):
    """Compute average perplexity over a set of sentences."""
    sentences = sentences[:max_samples]
    total_loss = 0.0
    total_tokens = 0

    for sentence in tqdm(sentences, desc="Evaluating perplexity"):
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model(tokens)

        shift_logits = logits[0, :-1, :]
        shift_labels = tokens[0, 1:]

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.shape[0]

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def evaluate_with_mean_ablation(model, sentences, edges, mean_activations,
                                 alpha=0.5, max_samples=200):
    """Compute perplexity with mean ablation applied.

    V2: Replaces activations with precomputed mean values instead of zero.
    Uses alpha blending for partial intervention.

    Args:
        model: HookedTransformer model.
        sentences: List of sentence strings.
        edges: List of Edge objects to ablate.
        mean_activations: Precomputed mean activations dict.
        alpha: Blending factor (0=no change, 1=full mean replacement).
        max_samples: Max sentences to evaluate.

    Returns:
        Average perplexity with mean ablation.
    """
    sentences = sentences[:max_samples]

    # Build mean-ablation hooks
    ablation_hooks = build_mean_ablation_hooks(edges, mean_activations, alpha=alpha)

    total_loss = 0.0
    total_tokens = 0

    for sentence in tqdm(sentences, desc=f"Evaluating perplexity (mean ablation, alpha={alpha})"):
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=ablation_hooks)

        shift_logits = logits[0, :-1, :]
        shift_labels = tokens[0, 1:]

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.shape[0]

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate collateral damage on CoLA")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name: gpt2, gpt2-medium, gpt2-large")
    parser.add_argument("--edges", type=str, default=None,
                        help="Path to top edges JSON")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max CoLA sentences to evaluate")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blending factor for mean ablation (default 0.5)")
    parser.add_argument("--output", type=str, default="results/cola_evaluation.json",
                        help="Output path for evaluation results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 4: COLLATERAL DAMAGE CHECK (CoLA) -- V2")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Edges:       {args.edges or 'None (baseline only)'}")
    print(f"  Alpha:       {args.alpha}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Output:      {args.output}")
    print("=" * 60)

    # Load model
    model = load_model(args.model, device=args.device)

    # Load CoLA dataset
    cola_data = load_cola_dataset(split="validation")
    sentences = [item["sentence"] for item in cola_data]

    # Baseline perplexity
    print("\n[CoLA] Computing baseline perplexity...")
    baseline_ppl = evaluate_perplexity(model, sentences, args.max_samples)
    print(f"[CoLA] Baseline perplexity: {baseline_ppl:.2f}")

    results = {
        "model": args.model,
        "n_samples": min(len(sentences), args.max_samples),
        "baseline_perplexity": baseline_ppl,
    }

    # Ablated perplexity with mean ablation
    if args.edges:
        edges = load_edges(args.edges)

        # Compute mean activations from neutral sentences
        print("\n[CoLA] Computing mean activations from neutral sentences...")
        mean_activations = compute_mean_activations(model)

        print(f"\n[CoLA] Computing perplexity with {len(edges)} edges (mean ablation, alpha={args.alpha})...")
        ablated_ppl = evaluate_with_mean_ablation(
            model, sentences, edges, mean_activations,
            alpha=args.alpha, max_samples=args.max_samples
        )
        print(f"[CoLA] Ablated perplexity: {ablated_ppl:.2f}")

        ppl_increase = ((ablated_ppl - baseline_ppl) / baseline_ppl * 100)
        results["ablated_perplexity"] = ablated_ppl
        results["perplexity_increase_percent"] = ppl_increase
        results["n_edges_ablated"] = len(edges)
        results["alpha"] = args.alpha
        results["ablation_method"] = "mean_ablation"

        print(f"\n{'=' * 60}")
        print(f"  COLLATERAL DAMAGE REPORT (V2)")
        print(f"{'=' * 60}")
        print(f"  Baseline perplexity:  {baseline_ppl:.2f}")
        print(f"  Ablated perplexity:   {ablated_ppl:.2f}")
        print(f"  Increase:             {ppl_increase:.1f}%")
        print(f"  Edges ablated:        {len(edges)}")
        print(f"  Alpha:                {args.alpha}")
        print(f"  Method:               Mean Ablation")
        if ppl_increase < 5:
            print(f"  Verdict:              [PASS] Minimal damage (< 5%)")
        elif ppl_increase < 15:
            print(f"  Verdict:              [WARN] Moderate damage (5-15%)")
        else:
            print(f"  Verdict:              [FAIL] Significant damage (> 15%)")
        print(f"{'=' * 60}")
    else:
        print("\n[CoLA] No edges provided, skipping ablation evaluation.")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] CoLA evaluation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
