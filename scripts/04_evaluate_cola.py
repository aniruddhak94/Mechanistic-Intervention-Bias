#!/usr/bin/env python3
"""
04_evaluate_cola.py — Check for collateral damage on CoLA.

Evaluates the model on the Corpus of Linguistic Acceptability (CoLA)
with and without edge ablation to ensure the model's grammar ability
is preserved after debiasing.

Usage:
    python scripts/04_evaluate_cola.py --model gpt2 --edges results/top_edges_gender.json

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
from src.eap_algorithm import load_edges, _attn_out_hook, _mlp_out_hook


def evaluate_perplexity(model, sentences, max_samples=200):
    """Compute average perplexity over a set of sentences.

    Lower perplexity = model assigns higher probability to the sentences
    = better language modeling ability.

    Args:
        model: HookedTransformer model.
        sentences: List of sentence strings.
        max_samples: Max number of sentences to evaluate (for speed).

    Returns:
        Average perplexity (float).
    """
    sentences = sentences[:max_samples]
    total_loss = 0.0
    total_tokens = 0

    for sentence in tqdm(sentences, desc="Evaluating perplexity"):
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model(tokens)

        # Compute cross-entropy loss (next-token prediction)
        # logits: [1, seq_len, vocab_size], targets: tokens shifted by 1
        shift_logits = logits[0, :-1, :]          # [seq_len-1, vocab_size]
        shift_labels = tokens[0, 1:]              # [seq_len-1]

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.shape[0]

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def evaluate_with_ablation(model, sentences, edges, max_samples=200):
    """Compute perplexity with edge ablation applied.

    For CoLA evaluation, we approximate ablation by zeroing out the
    activations at destination hooks of the top edges (since we don't
    have clean/corrupted pairs for CoLA sentences).

    Args:
        model: HookedTransformer model.
        sentences: List of sentence strings.
        edges: List of Edge objects to ablate.
        max_samples: Max sentences to evaluate.

    Returns:
        Average perplexity with ablation.
    """
    sentences = sentences[:max_samples]

    # Build zero-ablation hooks (set dst activations to zero for patched heads)
    n_heads = model.cfg.n_heads
    dst_groups = {}
    for edge in edges:
        if edge.dst_type == "attn":
            hook_name = _attn_out_hook(edge.dst_layer)
        else:
            hook_name = _mlp_out_hook(edge.dst_layer)
        if hook_name not in dst_groups:
            dst_groups[hook_name] = {"type": edge.dst_type, "heads": set()}
        if edge.dst_head is not None:
            dst_groups[hook_name]["heads"].add(edge.dst_head)

    def make_ablation_hooks():
        hooks = []
        for hook_name, info in dst_groups.items():
            if info["type"] == "attn" and info["heads"]:
                heads = list(info["heads"])
                def make_hook(h_list):
                    def hook_fn(activation, hook):
                        for h in h_list:
                            activation[:, :, h, :] = 0.0
                        return activation
                    return hook_fn
                hooks.append((hook_name, make_hook(heads)))
            elif info["type"] == "mlp":
                def make_mlp_hook():
                    def hook_fn(activation, hook):
                        return torch.zeros_like(activation)
                    return hook_fn
                hooks.append((hook_name, make_mlp_hook()))
        return hooks

    ablation_hooks = make_ablation_hooks()

    total_loss = 0.0
    total_tokens = 0

    for sentence in tqdm(sentences, desc="Evaluating perplexity (ablated)"):
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
                        help="Path to top edges JSON (optional, skip ablation eval if not provided)")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max CoLA sentences to evaluate (for speed)")
    parser.add_argument("--output", type=str, default="results/cola_evaluation.json",
                        help="Output path for evaluation results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 4: COLLATERAL DAMAGE CHECK (CoLA)")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Edges:       {args.edges or 'None (baseline only)'}")
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

    # Ablated perplexity
    if args.edges:
        edges = load_edges(args.edges)
        print(f"\n[CoLA] Computing perplexity with {len(edges)} edges ablated...")
        ablated_ppl = evaluate_with_ablation(model, sentences, edges, args.max_samples)
        print(f"[CoLA] Ablated perplexity: {ablated_ppl:.2f}")

        ppl_increase = ((ablated_ppl - baseline_ppl) / baseline_ppl * 100)
        results["ablated_perplexity"] = ablated_ppl
        results["perplexity_increase_percent"] = ppl_increase
        results["n_edges_ablated"] = len(edges)

        print(f"\n{'=' * 60}")
        print(f"  COLLATERAL DAMAGE REPORT")
        print(f"{'=' * 60}")
        print(f"  Baseline perplexity:  {baseline_ppl:.2f}")
        print(f"  Ablated perplexity:   {ablated_ppl:.2f}")
        print(f"  Increase:             {ppl_increase:.1f}%")
        print(f"  Edges ablated:        {len(edges)}")
        if ppl_increase < 5:
            print(f"  Verdict:              ✅ Minimal damage (< 5%)")
        elif ppl_increase < 15:
            print(f"  Verdict:              ⚠️  Moderate damage (5-15%)")
        else:
            print(f"  Verdict:              ❌ Significant damage (> 15%)")
        print(f"{'=' * 60}")
    else:
        print("\n[CoLA] No edges provided, skipping ablation evaluation.")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ CoLA evaluation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
