"""
intervention.py — Activation patching for debiasing.

Given a set of top bias-causing edges (from EAP), this module patches
those edges during inference: it replaces their clean activations with
the corresponding corrupted (neutral) activations.  This "turns off"
the bias circuit without retraining the model.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.eap_algorithm import Edge, _attn_out_hook, _mlp_out_hook


def _build_patch_hooks(
    model,
    edges: List[Edge],
    corrupted_cache,
) -> list:
    """Build forward hooks that patch specified edges with corrupted activations.

    For each destination component that has at least one incoming edge to patch,
    we create a hook that modifies the activation in-place by blending in the
    corrupted-run activation for the relevant sub-components (specific heads, etc).

    Args:
        model: HookedTransformer model.
        edges: List of Edge objects to patch.
        corrupted_cache: ActivationCache from the corrupted forward pass.

    Returns:
        List of (hook_name, hook_fn) tuples for model.run_with_hooks.
    """
    n_heads = model.cfg.n_heads

    # Group edges by destination hook point
    dst_groups: Dict[str, List[Edge]] = {}
    for edge in edges:
        if edge.dst_type == "attn":
            hook_name = _attn_out_hook(edge.dst_layer)
        else:
            hook_name = _mlp_out_hook(edge.dst_layer)

        if hook_name not in dst_groups:
            dst_groups[hook_name] = []
        dst_groups[hook_name].append(edge)

    hooks = []

    for hook_name, group_edges in dst_groups.items():
        # Determine which heads (if attention) to patch
        if group_edges[0].dst_type == "attn":
            heads_to_patch = set()
            for e in group_edges:
                if e.dst_head is not None:
                    heads_to_patch.add(e.dst_head)

            def make_attn_hook(hname, heads):
                def hook_fn(activation, hook):
                    corrupted_act = corrupted_cache[hname]
                    for head in heads:
                        activation[:, :, head, :] = corrupted_act[:, :, head, :]
                    return activation
                return hook_fn

            hooks.append((hook_name, make_attn_hook(hook_name, heads_to_patch)))

        else:  # MLP
            def make_mlp_hook(hname):
                def hook_fn(activation, hook):
                    corrupted_act = corrupted_cache[hname]
                    return corrupted_act
                return hook_fn

            hooks.append((hook_name, make_mlp_hook(hook_name)))

    return hooks


def ablate_edges(
    model,
    edges: List[Edge],
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Run the model with specified edges patched (ablated).

    Replaces activations at edge destinations with corrupted-run activations,
    effectively "turning off" those edges' contribution to bias.

    Args:
        model: HookedTransformer model.
        edges: List of Edge objects to ablate.
        clean_tokens: Tokenized clean prompt [1, seq_len].
        corrupted_tokens: Tokenized corrupted prompt [1, seq_len].

    Returns:
        Logits tensor [1, seq_len, vocab_size] from the patched forward pass.
    """
    device = model.cfg.device
    clean_tokens = clean_tokens.to(device)
    corrupted_tokens = corrupted_tokens.to(device)

    # Get corrupted activations
    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # Build patching hooks
    patch_hooks = _build_patch_hooks(model, edges, corrupted_cache)

    # Run clean tokens through model with patches applied
    with torch.no_grad():
        logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=patch_hooks,
        )

    return logits


def measure_debiased_score(
    model,
    edges: List[Edge],
    prompt_pairs: List[Dict],
    male_ids: List[int],
    female_ids: List[int],
) -> Dict:
    """Compute bias score after intervention for all prompt pairs.

    Args:
        model: HookedTransformer model.
        edges: List of Edge objects to ablate.
        prompt_pairs: List of dicts from create_prompt_pairs.
        male_ids: Token IDs for male-gendered words.
        female_ids: Token IDs for female-gendered words.

    Returns:
        Dict with per-prompt results and aggregate statistics.
    """
    results = []
    min_len = min(len(male_ids), len(female_ids))

    for pair in tqdm(prompt_pairs, desc="Measuring debiased scores"):
        logits = ablate_edges(
            model, edges,
            pair["clean_tokens"],
            pair["corrupted_tokens"],
        )

        last_logits = logits[0, -1, :]
        log_probs = F.log_softmax(last_logits, dim=-1)
        probs = F.softmax(last_logits, dim=-1)

        male_lp = log_probs[male_ids[:min_len]]
        female_lp = log_probs[female_ids[:min_len]]

        bias_score = torch.norm(male_lp - female_lp, p=2).item()
        male_prob = probs[male_ids].sum().item()
        female_prob = probs[female_ids].sum().item()

        results.append({
            "id": pair["id"],
            "clean_prompt": pair["clean_prompt"],
            "bias_score_after": bias_score,
            "male_prob_after": male_prob,
            "female_prob_after": female_prob,
            "direction_after": "male" if male_prob > female_prob else "female",
        })

    scores = [r["bias_score_after"] for r in results]
    summary = {
        "per_prompt": results,
        "mean_bias_after": sum(scores) / len(scores),
        "max_bias_after": max(scores),
        "min_bias_after": min(scores),
        "n_prompts": len(scores),
        "n_edges_ablated": len(edges),
    }

    print(f"\n[intervention] Mean bias after: {summary['mean_bias_after']:.4f}")
    print(f"[intervention] Edges ablated: {len(edges)}")

    return summary


def run_debiasing_comparison(
    model,
    edges: List[Edge],
    prompt_pairs: List[Dict],
    male_ids: List[int],
    female_ids: List[int],
    baseline_results: Dict,
    save_path: str = None,
) -> Dict:
    """Run full debiasing pipeline and compare before/after.

    Args:
        model: HookedTransformer.
        edges: Edges to ablate.
        prompt_pairs: Tokenized prompt pairs.
        male_ids: Male token IDs.
        female_ids: Female token IDs.
        baseline_results: Dict from run_baseline.
        save_path: Optional path to save comparison results.

    Returns:
        Dict with before, after, and reduction statistics.
    """
    after = measure_debiased_score(model, edges, prompt_pairs, male_ids, female_ids)

    before_mean = baseline_results["mean_bias"]
    after_mean = after["mean_bias_after"]
    reduction = ((before_mean - after_mean) / before_mean * 100) if before_mean > 0 else 0

    comparison = {
        "before": {
            "mean_bias": before_mean,
            "max_bias": baseline_results["max_bias"],
            "min_bias": baseline_results["min_bias"],
        },
        "after": {
            "mean_bias": after_mean,
            "max_bias": after["max_bias_after"],
            "min_bias": after["min_bias_after"],
        },
        "reduction_percent": reduction,
        "n_edges_ablated": len(edges),
        "per_prompt": [],
    }

    # Combine per-prompt before/after
    for before_p, after_p in zip(baseline_results["per_prompt"], after["per_prompt"]):
        combined = {
            "id": before_p["id"],
            "clean_prompt": before_p["clean_prompt"],
            "bias_before": before_p["bias_score"],
            "bias_after": after_p["bias_score_after"],
            "reduction": before_p["bias_score"] - after_p["bias_score_after"],
        }
        comparison["per_prompt"].append(combined)

    print(f"\n{'=' * 60}")
    print(f"  DEBIASING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Mean bias BEFORE: {before_mean:.4f}")
    print(f"  Mean bias AFTER:  {after_mean:.4f}")
    print(f"  Reduction:        {reduction:.1f}%")
    print(f"  Edges ablated:    {len(edges)}")
    print(f"{'=' * 60}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"[intervention] Results saved to {save_path}")

    return comparison
