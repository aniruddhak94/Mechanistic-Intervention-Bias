#!/usr/bin/env python3
"""
06_pareto_sweep.py -- Systematic alpha sweep for optimal tradeoff.

Tests multiple alpha values and records bias reduction + perplexity
at each point to plot the Pareto frontier.

Usage (Kaggle):
    python scripts/06_pareto_sweep.py --model gpt2 --dataset data/gender_bias.json

Output:
    results/pareto_sweep.json
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
from src.data_utils import (
    load_bias_dataset, create_prompt_pairs,
    get_gendered_token_ids, load_cola_dataset,
)
from src.eap_algorithm import load_edges
from src.intervention import (
    compute_mean_activations, build_mean_ablation_hooks,
    measure_debiased_score,
)
from src.baseline_scoring import run_baseline


def evaluate_perplexity_with_hooks(model, sentences, hooks, max_samples=100):
    """Compute perplexity with intervention hooks applied."""
    sentences = sentences[:max_samples]
    total_loss, total_tokens = 0.0, 0

    for sentence in sentences:
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        shift_logits = logits[0, :-1, :]
        shift_labels = tokens[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.shape[0]

    avg_loss = total_loss / max(total_tokens, 1)
    return torch.exp(torch.tensor(avg_loss)).item()


def main():
    parser = argparse.ArgumentParser(description="Alpha sweep for Pareto frontier")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="data/gender_bias.json")
    parser.add_argument("--edges", type=str, default="results/top_edges_gender.json")
    parser.add_argument("--alphas", type=str,
                        default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,1.0")
    parser.add_argument("--cola_samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="results/pareto_sweep.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]

    print("=" * 60)
    print("  STEP 6: PARETO ALPHA SWEEP")
    print("=" * 60)
    print(f"  Alphas:  {alphas}")
    print(f"  Output:  {args.output}")
    print("=" * 60)

    model = load_model(args.model, device=args.device)
    bias_data = load_bias_dataset(args.dataset)
    prompt_pairs = create_prompt_pairs(bias_data, model)
    male_ids, female_ids = get_gendered_token_ids(model)
    edges = load_edges(args.edges)

    # Baseline
    baseline = run_baseline(model, prompt_pairs, male_ids, female_ids)
    baseline_bias = baseline["mean_bias"]

    # CoLA
    cola_data = load_cola_dataset(split="validation")
    cola_sentences = [item["sentence"] for item in cola_data]

    # Baseline perplexity
    baseline_ppl = evaluate_perplexity_with_hooks(model, cola_sentences, [], args.cola_samples)
    print(f"\n[Sweep] Baseline bias: {baseline_bias:.4f}")
    print(f"[Sweep] Baseline PPL:  {baseline_ppl:.2f}\n")

    # Precompute mean activations
    mean_activations = compute_mean_activations(model)

    results = []
    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")

        # Bias measurement
        after = measure_debiased_score(
            model, edges, prompt_pairs, male_ids, female_ids, alpha=alpha,
        )
        after_bias = after["mean_bias_after"]
        reduction = (baseline_bias - after_bias) / baseline_bias * 100

        # Perplexity with mean ablation
        hooks = build_mean_ablation_hooks(edges, mean_activations, alpha=alpha)
        ablated_ppl = evaluate_perplexity_with_hooks(
            model, cola_sentences, hooks, args.cola_samples,
        )
        ppl_increase = (ablated_ppl - baseline_ppl) / baseline_ppl * 100

        # Count improved prompts
        per_prompt = after.get("per_prompt", [])
        n_prompts = len(per_prompt)
        # We need before scores to count improved
        improved = 0
        for pp, bp in zip(per_prompt, baseline.get("per_prompt", [])):
            if pp["bias_score_after"] < bp["bias_score"]:
                improved += 1
        improved_pct = improved / max(n_prompts, 1) * 100

        result = {
            "alpha": alpha,
            "bias_before": round(baseline_bias, 4),
            "bias_after": round(after_bias, 4),
            "bias_reduction_percent": round(reduction, 2),
            "baseline_perplexity": round(baseline_ppl, 2),
            "ablated_perplexity": round(ablated_ppl, 2),
            "perplexity_increase_percent": round(ppl_increase, 2),
            "prompts_improved": improved,
            "prompts_improved_percent": round(improved_pct, 1),
            "n_edges": len(edges),
        }
        results.append(result)

        print(f"  Bias reduction: {reduction:.2f}%")
        print(f"  PPL increase:   {ppl_increase:.2f}%")
        print(f"  Prompts improved: {improved}/{n_prompts} ({improved_pct:.0f}%)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {"sweep_results": results, "config": {"alphas": alphas, "n_edges": len(edges)}}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Pareto sweep results saved to {args.output}")


if __name__ == "__main__":
    main()
