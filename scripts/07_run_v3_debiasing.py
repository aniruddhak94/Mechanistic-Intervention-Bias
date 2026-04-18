#!/usr/bin/env python3
"""
07_run_v3_debiasing.py -- V3 debiasing with all three methods.

Runs SAE Feature Ablation, CAA Steering, and LEACE Projection,
then evaluates each on bias reduction and perplexity.

Usage (Kaggle):
    python scripts/07_run_v3_debiasing.py --model gpt2 --dataset data/gender_bias.json

Output:
    results/v3_debiasing_results.json
    results/v3_probe_results.json
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
from src.baseline_scoring import run_baseline
from src.linear_probe import (
    run_probing_all_layers,
    compute_caa_steering_vectors, build_caa_hooks,
    build_leace_hooks,
)


def evaluate_perplexity(model, sentences, max_samples=200):
    """Baseline perplexity (no hooks)."""
    sentences = sentences[:max_samples]
    total_loss, total_tokens = 0.0, 0
    for sentence in tqdm(sentences, desc="Baseline PPL"):
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2: continue
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
    for sentence in tqdm(sentences, desc="Ablated PPL"):
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2: continue
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

    for pair in tqdm(prompt_pairs, desc="Measuring bias"):
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


def run_method(name, model, prompt_pairs, male_ids, female_ids,
               baseline_results, cola_sentences, hooks, n_targets):
    """Run a single debiasing method and return all metrics."""
    print(f"\n{'='*50}")
    print(f"  Running: {name}")
    print(f"{'='*50}")

    # Bias after intervention
    after = measure_bias_with_hooks(model, prompt_pairs, male_ids, female_ids, hooks)

    before_mean = baseline_results["mean_bias"]
    after_mean = after["mean_bias"]
    reduction = (before_mean - after_mean) / before_mean * 100

    # Count improved
    improved = 0
    per_prompt_combined = []
    for bp, ap in zip(baseline_results["per_prompt"], after["per_prompt"]):
        imp = ap["bias_score"] < bp["bias_score"]
        if imp: improved += 1
        per_prompt_combined.append({
            "id": bp["id"],
            "clean_prompt": bp["clean_prompt"],
            "bias_before": bp["bias_score"],
            "bias_after": ap["bias_score"],
            "reduction": bp["bias_score"] - ap["bias_score"],
            "improved": imp,
        })

    n_prompts = len(per_prompt_combined)
    improved_pct = improved / max(n_prompts, 1) * 100

    # Perplexity
    baseline_ppl = evaluate_perplexity(model, cola_sentences)
    ablated_ppl = evaluate_perplexity_with_hooks(model, cola_sentences, hooks)
    ppl_increase = (ablated_ppl - baseline_ppl) / baseline_ppl * 100

    result = {
        "method": name,
        "bias_before": round(before_mean, 4),
        "bias_after": round(after_mean, 4),
        "bias_reduction_percent": round(reduction, 2),
        "prompts_improved": improved,
        "prompts_improved_percent": round(improved_pct, 1),
        "n_prompts": n_prompts,
        "baseline_perplexity": round(baseline_ppl, 2),
        "ablated_perplexity": round(ablated_ppl, 2),
        "perplexity_increase_percent": round(ppl_increase, 2),
        "n_targets": n_targets,
        "per_prompt": per_prompt_combined,
    }

    print(f"  Bias: {before_mean:.4f} -> {after_mean:.4f} ({reduction:+.2f}%)")
    print(f"  Prompts improved: {improved}/{n_prompts} ({improved_pct:.0f}%)")
    print(f"  PPL: {baseline_ppl:.2f} -> {ablated_ppl:.2f} ({ppl_increase:+.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description="V3 debiasing: SAE + CAA + LEACE")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="data/gender_bias.json")
    parser.add_argument("--features", type=str, default="results/gender_features.json",
                        help="Gender features from 05_sae_discovery.py")
    parser.add_argument("--layers", type=str, default="6,8,9,10,11")
    parser.add_argument("--caa_strength", type=float, default=3.0)
    parser.add_argument("--leace_alpha", type=float, default=1.0)
    parser.add_argument("--sae_alpha", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="results/v3_debiasing_results.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print("=" * 60)
    print("  STEP 7: V3 DEBIASING -- ALL METHODS")
    print("=" * 60)

    model = load_model(args.model, device=args.device)
    bias_data = load_bias_dataset(args.dataset)
    prompt_pairs = create_prompt_pairs(bias_data, model)
    male_ids, female_ids = get_gendered_token_ids(model)

    # CoLA sentences
    cola_data = load_cola_dataset(split="validation")
    cola_sentences = [item["sentence"] for item in cola_data]

    # Baseline
    print("\n[V3] Computing baseline...")
    baseline = run_baseline(model, prompt_pairs, male_ids, female_ids)

    # ─── Linear Probing (for probe accuracy plot + CAA + LEACE) ───
    print("\n[V3] Running linear probes...")
    probe_results = run_probing_all_layers(model, prompt_pairs, layers=list(range(12)))

    # Save probe results
    probe_path = args.output.replace("debiasing_results", "probe_results")
    os.makedirs(os.path.dirname(probe_path), exist_ok=True)
    with open(probe_path, "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)
    print(f"[V3] Probe results saved to {probe_path}")

    all_method_results = {}

    # ─── Method 1: CAA Steering ───
    print("\n[V3] Computing CAA steering vectors...")
    steering_vecs = compute_caa_steering_vectors(model, prompt_pairs, layers=layers)
    caa_hooks = build_caa_hooks(steering_vecs, strength=args.caa_strength,
                                 device=model.cfg.device)
    caa_result = run_method(
        "CAA Steering", model, prompt_pairs, male_ids, female_ids,
        baseline, cola_sentences, caa_hooks, n_targets=len(layers),
    )
    all_method_results["CAA Steering"] = caa_result

    # ─── Method 2: LEACE Projection ───
    print("\n[V3] Computing LEACE projections...")
    leace_hooks = build_leace_hooks(model, prompt_pairs, layers=layers,
                                     alpha=args.leace_alpha)
    leace_result = run_method(
        "LEACE Projection", model, prompt_pairs, male_ids, female_ids,
        baseline, cola_sentences, leace_hooks, n_targets=len(layers),
    )
    all_method_results["LEACE Projection"] = leace_result

    # ─── Method 3: SAE Feature Ablation ───
    if os.path.exists(args.features):
        with open(args.features, "r") as f:
            gender_features = json.load(f)

        from src.sae_analysis import build_feature_ablation_hooks
        sae_hooks = build_feature_ablation_hooks(
            model, gender_features, layers=layers, alpha=args.sae_alpha,
        )
        n_feats = sum(
            len(gender_features["features_per_layer"].get(f"layer_{l}", []))
            for l in layers
        )
        sae_result = run_method(
            "SAE Feature Ablation", model, prompt_pairs, male_ids, female_ids,
            baseline, cola_sentences, sae_hooks, n_targets=n_feats,
        )
        all_method_results["SAE Feature Ablation"] = sae_result
    else:
        print(f"\n[WARN] {args.features} not found. Skipping SAE method.")
        print(f"       Run 05_sae_discovery.py first.")

    # ─── Save all results ───
    output = {
        "baseline": {
            "mean_bias": baseline["mean_bias"],
            "max_bias": baseline["max_bias"],
            "min_bias": baseline["min_bias"],
        },
        "methods": all_method_results,
        "probe_results": probe_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        # Remove per_prompt from methods for cleaner JSON (keep separate)
        output_clean = json.loads(json.dumps(output, default=str))
        json.dump(output_clean, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  V3 RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, res in all_method_results.items():
        print(f"  {name}:")
        print(f"    Bias Reduction:  {res['bias_reduction_percent']:.1f}%")
        print(f"    PPL Increase:    {res['perplexity_increase_percent']:.1f}%")
        print(f"    Prompts Improved:{res['prompts_improved_percent']:.0f}%")
    print(f"{'='*60}")
    print(f"\n[OK] V3 results saved to {args.output}")


if __name__ == "__main__":
    main()
