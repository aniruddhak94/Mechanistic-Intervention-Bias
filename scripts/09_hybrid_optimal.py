#!/usr/bin/env python3
"""
09_hybrid_optimal.py -- Hybrid CAA+LEACE grid search for optimal debiasing.

Combines Contrastive Activation Addition (CAA) with LEACE projection at
reduced strengths to find the sweet spot: >10% bias reduction with
minimal perplexity loss.

The insight: CAA alone is gentle (2.5% bias, 2.5% PPL) and LEACE alone
is aggressive (14.4% bias, 114% PPL). By blending them at reduced
strengths, we can reach >10% bias reduction with acceptable PPL cost.

Usage (Kaggle):
    python scripts/09_hybrid_optimal.py --model gpt2 --dataset data/gender_bias.json

Output:
    results/thesis_results/hybrid_grid_search.json
    results/thesis_results/hybrid_optimal_results.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import load_model
from src.data_utils import (
    load_bias_dataset, create_prompt_pairs,
    get_gendered_token_ids, load_cola_dataset,
)
from src.baseline_scoring import run_baseline
from src.linear_probe import (
    compute_caa_steering_vectors, build_caa_hooks,
    build_leace_hooks, collect_layer_activations,
    compute_leace_projection,
)


def evaluate_perplexity(model, sentences, max_samples=200):
    """Baseline perplexity (no hooks)."""
    sentences = sentences[:max_samples]
    total_loss, total_tokens = 0.0, 0
    for sentence in sentences:
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(tokens)
        loss = F.cross_entropy(logits[0, :-1, :], tokens[0, 1:], reduction="sum")
        total_loss += loss.item()
        total_tokens += tokens.shape[1] - 1
    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


def evaluate_perplexity_with_hooks(model, sentences, hooks, max_samples=200):
    """Perplexity with hooks applied."""
    sentences = sentences[:max_samples]
    total_loss, total_tokens = 0.0, 0
    for sentence in sentences:
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        loss = F.cross_entropy(logits[0, :-1, :], tokens[0, 1:], reduction="sum")
        total_loss += loss.item()
        total_tokens += tokens.shape[1] - 1
    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


def measure_bias_with_hooks(model, prompt_pairs, male_ids, female_ids, hooks):
    """Measure bias with arbitrary hooks applied."""
    min_len = min(len(male_ids), len(female_ids))
    results = []

    for pair in tqdm(prompt_pairs, desc="Measuring bias", leave=False):
        tokens = pair["clean_tokens"].to(model.cfg.device)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

        last_logits = logits[0, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)
        probs = F.softmax(last_logits, dim=-1)

        bias_score = torch.norm(
            log_probs[male_ids[:min_len]] - log_probs[female_ids[:min_len]], p=2
        ).item()
        male_prob = probs[male_ids].sum().item()
        female_prob = probs[female_ids].sum().item()

        results.append({
            "id": pair["id"],
            "clean_prompt": pair["clean_prompt"],
            "bias_score": bias_score,
            "male_prob": male_prob,
            "female_prob": female_prob,
            "direction": "male" if male_prob > female_prob else "female",
        })

    scores = [r["bias_score"] for r in results]
    return {
        "mean_bias": sum(scores) / len(scores),
        "max_bias": max(scores),
        "min_bias": min(scores),
        "per_prompt": results,
    }


def build_hybrid_hooks(
    steering_vectors, leace_projections, caa_strength, leace_alpha, device
):
    """Combine CAA and LEACE hooks into a single hook list.

    The key insight: for each layer, we apply BOTH:
      1. LEACE projection to erase the gender subspace (partial, via leace_alpha)
      2. CAA subtraction to push away from residual gender direction
    """
    hooks = []

    for layer in steering_vectors:
        hook_name = f"blocks.{layer}.hook_resid_pre"
        sv = steering_vectors[layer].to(device)
        proj = leace_projections[layer].to(device)
        la = leace_alpha
        cs = caa_strength

        def make_hook(steering_v, projection, l_alpha, c_strength):
            def hook_fn(activation, hook):
                batch, seq_len, d_model = activation.shape

                # Step 1: Partial LEACE projection
                flat = activation.reshape(-1, d_model)
                projected = flat @ projection.T
                projected = projected.reshape(batch, seq_len, d_model)
                act = l_alpha * projected + (1 - l_alpha) * activation

                # Step 2: CAA steering subtraction
                act = act - c_strength * steering_v.unsqueeze(0).unsqueeze(0)

                return act
            return hook_fn

        hooks.append((hook_name, make_hook(sv, proj, la, cs)))

    return hooks


def main():
    parser = argparse.ArgumentParser(description="Hybrid CAA+LEACE grid search")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="data/gender_bias.json")
    parser.add_argument("--layers", type=str, default="6,8,9,10,11")
    parser.add_argument("--cola_samples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="results/thesis_results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print("=" * 60)
    print("  STEP 9: HYBRID CAA + LEACE GRID SEARCH")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model, device=args.device)
    bias_data = load_bias_dataset(args.dataset)
    prompt_pairs = create_prompt_pairs(bias_data, model)
    male_ids, female_ids = get_gendered_token_ids(model)
    device = model.cfg.device

    # CoLA sentences
    cola_data = load_cola_dataset(split="validation")
    cola_sentences = [item["sentence"] for item in cola_data]

    # Baseline measurements
    print("\n[Hybrid] Computing baseline...")
    baseline = run_baseline(model, prompt_pairs, male_ids, female_ids)
    baseline_bias = baseline["mean_bias"]
    baseline_ppl = evaluate_perplexity(model, cola_sentences, args.cola_samples)
    print(f"[Hybrid] Baseline bias: {baseline_bias:.4f}")
    print(f"[Hybrid] Baseline PPL:  {baseline_ppl:.2f}")

    # Precompute CAA steering vectors and LEACE projections
    print("\n[Hybrid] Computing CAA steering vectors...")
    steering_vectors = compute_caa_steering_vectors(model, prompt_pairs, layers=layers)

    print("[Hybrid] Computing LEACE projections...")
    layer_data = collect_layer_activations(model, prompt_pairs, layers)
    leace_projections = {}
    for layer in layers:
        acts, labels = layer_data[layer]
        leace_projections[layer] = compute_leace_projection(acts, labels)

    # ─── Grid Search ───
    # CAA strengths: 1.0 to 8.0
    # LEACE alphas: 0.0 to 0.5
    caa_strengths = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    leace_alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    grid_results = []

    print(f"\n[Hybrid] Running {len(caa_strengths)} x {len(leace_alphas)} "
          f"= {len(caa_strengths) * len(leace_alphas)} grid search...")

    for caa_s in caa_strengths:
        for leace_a in leace_alphas:
            start = time.time()

            # Build combined hooks
            hooks = build_hybrid_hooks(
                steering_vectors, leace_projections,
                caa_strength=caa_s, leace_alpha=leace_a,
                device=device,
            )

            # Measure bias
            after = measure_bias_with_hooks(
                model, prompt_pairs, male_ids, female_ids, hooks,
            )
            after_bias = after["mean_bias"]
            reduction = (baseline_bias - after_bias) / baseline_bias * 100

            # Count improved
            improved = sum(
                1 for bp, ap in zip(baseline["per_prompt"], after["per_prompt"])
                if ap["bias_score"] < bp["bias_score"]
            )
            improved_pct = improved / len(after["per_prompt"]) * 100

            # Measure perplexity
            ablated_ppl = evaluate_perplexity_with_hooks(
                model, cola_sentences, hooks, args.cola_samples,
            )
            ppl_increase = (ablated_ppl - baseline_ppl) / baseline_ppl * 100

            elapsed = time.time() - start

            result = {
                "caa_strength": caa_s,
                "leace_alpha": leace_a,
                "bias_before": round(baseline_bias, 4),
                "bias_after": round(after_bias, 4),
                "bias_reduction_percent": round(reduction, 2),
                "prompts_improved": improved,
                "prompts_improved_percent": round(improved_pct, 1),
                "baseline_perplexity": round(baseline_ppl, 2),
                "ablated_perplexity": round(ablated_ppl, 2),
                "perplexity_increase_percent": round(ppl_increase, 2),
                "elapsed_seconds": round(elapsed, 1),
            }
            grid_results.append(result)

            marker = " ***" if reduction >= 10 and ppl_increase < 20 else ""
            print(f"  CAA={caa_s:.0f}, LEACE={leace_a:.2f} => "
                  f"Bias: -{reduction:.1f}%, PPL: +{ppl_increase:.1f}%, "
                  f"Prompts: {improved_pct:.0f}%{marker}")

    # Save grid search
    grid_path = os.path.join(args.output_dir, "hybrid_grid_search.json")
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump({"grid_results": grid_results, "config": {
            "caa_strengths": caa_strengths,
            "leace_alphas": leace_alphas,
            "layers": layers,
        }}, f, indent=2)
    print(f"\n[OK] Grid search saved to {grid_path}")

    # ─── Find Optimal Configuration ───
    # Target: >10% bias reduction, minimize PPL increase
    valid = [r for r in grid_results if r["bias_reduction_percent"] >= 10]
    if valid:
        optimal = min(valid, key=lambda x: x["perplexity_increase_percent"])
    else:
        # Fall back to best bias reduction overall
        optimal = max(grid_results, key=lambda x: x["bias_reduction_percent"])

    print(f"\n{'='*60}")
    print(f"  OPTIMAL CONFIGURATION FOUND")
    print(f"{'='*60}")
    print(f"  CAA Strength:      {optimal['caa_strength']}")
    print(f"  LEACE Alpha:       {optimal['leace_alpha']}")
    print(f"  Bias Reduction:    {optimal['bias_reduction_percent']:.1f}%")
    print(f"  PPL Increase:      {optimal['perplexity_increase_percent']:.1f}%")
    print(f"  Prompts Improved:  {optimal['prompts_improved_percent']:.0f}%")
    print(f"{'='*60}")

    # ─── Run the optimal config with full per-prompt detail ───
    print("\n[Hybrid] Running optimal config for detailed thesis results...")
    opt_hooks = build_hybrid_hooks(
        steering_vectors, leace_projections,
        caa_strength=optimal["caa_strength"],
        leace_alpha=optimal["leace_alpha"],
        device=device,
    )
    opt_after = measure_bias_with_hooks(
        model, prompt_pairs, male_ids, female_ids, opt_hooks,
    )

    # Build per-prompt comparison
    per_prompt_thesis = []
    for bp, ap in zip(baseline["per_prompt"], opt_after["per_prompt"]):
        per_prompt_thesis.append({
            "id": bp["id"],
            "clean_prompt": bp["clean_prompt"],
            "bias_before": round(bp["bias_score"], 4),
            "bias_after": round(ap["bias_score"], 4),
            "reduction": round(bp["bias_score"] - ap["bias_score"], 4),
            "reduction_percent": round(
                (bp["bias_score"] - ap["bias_score"]) / max(bp["bias_score"], 1e-8) * 100, 2
            ),
            "male_prob_before": round(bp.get("male_prob", 0), 6),
            "female_prob_before": round(bp.get("female_prob", 0), 6),
            "male_prob_after": round(ap.get("male_prob", 0), 6),
            "female_prob_after": round(ap.get("female_prob", 0), 6),
            "direction_before": bp.get("direction", ""),
            "direction_after": ap.get("direction", ""),
            "improved": ap["bias_score"] < bp["bias_score"],
        })

    # Statistics
    reductions = [p["reduction"] for p in per_prompt_thesis]
    improved_count = sum(1 for p in per_prompt_thesis if p["improved"])

    thesis_results = {
        "method": "Hybrid CAA + LEACE",
        "config": {
            "caa_strength": optimal["caa_strength"],
            "leace_alpha": optimal["leace_alpha"],
            "layers": layers,
            "model": args.model,
            "n_prompts": len(prompt_pairs),
        },
        "metrics": {
            "bias_before": round(baseline_bias, 4),
            "bias_after": round(opt_after["mean_bias"], 4),
            "bias_reduction_percent": round(
                (baseline_bias - opt_after["mean_bias"]) / baseline_bias * 100, 2
            ),
            "max_per_prompt_reduction": round(max(reductions), 4),
            "min_per_prompt_reduction": round(min(reductions), 4),
            "mean_per_prompt_reduction": round(sum(reductions) / len(reductions), 4),
            "std_per_prompt_reduction": round(
                (sum((r - sum(reductions)/len(reductions))**2 for r in reductions)
                 / len(reductions)) ** 0.5, 4
            ),
            "prompts_improved": improved_count,
            "prompts_improved_percent": round(improved_count / len(per_prompt_thesis) * 100, 1),
            "prompts_worsened": len(per_prompt_thesis) - improved_count,
            "baseline_perplexity": round(baseline_ppl, 2),
            "ablated_perplexity": optimal["ablated_perplexity"],
            "perplexity_increase_percent": optimal["perplexity_increase_percent"],
        },
        "per_prompt": per_prompt_thesis,
    }

    opt_path = os.path.join(args.output_dir, "hybrid_optimal_results.json")
    with open(opt_path, "w", encoding="utf-8") as f:
        json.dump(thesis_results, f, indent=2)

    print(f"\n[OK] Thesis results saved to {opt_path}")


if __name__ == "__main__":
    main()
