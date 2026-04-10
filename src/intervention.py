"""
intervention.py -- Activation patching for debiasing.

Given a set of top bias-causing edges (from EAP), this module patches
those edges during inference: it replaces their clean activations with
the corresponding corrupted (neutral) activations.  This "turns off"
the bias circuit without retraining the model.

V2 improvements:
  - Alpha blending: new_act = alpha * corrupted + (1 - alpha) * clean
  - Mean ablation: replace with precomputed mean activations (for CoLA eval)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.eap_algorithm import Edge, _attn_out_hook, _mlp_out_hook


# -- Mean activation computation -----------------------------------------------

NEUTRAL_SENTENCES = [
    "The sun rose over the mountains and the valley below.",
    "A gentle breeze rustled through the autumn leaves.",
    "The river flowed steadily toward the distant ocean.",
    "Birds sang in the trees as morning light filled the sky.",
    "The old bridge connected two sides of the quiet town.",
    "Rain fell softly on the cobblestone streets at dusk.",
    "Waves crashed against the rocky shore under gray clouds.",
    "The garden was filled with colorful flowers in spring.",
    "Snow covered the rooftops and the empty playground.",
    "A narrow path wound through the dense forest.",
    "The clock tower struck noon and pigeons scattered.",
    "Fireflies lit up the meadow on a warm summer night.",
    "The library was quiet except for the turning of pages.",
    "A steam train chugged along the countryside tracks.",
    "The market was bustling with vendors and shoppers.",
    "Stars appeared one by one as twilight deepened.",
    "The cat slept on the windowsill in a patch of sunlight.",
    "A fishing boat rocked gently in the harbor.",
    "The road stretched endlessly across the flat plains.",
    "Candles flickered in the windows of the old cottage.",
    "The lighthouse beam swept across the dark water.",
    "A kite soared high above the children in the park.",
    "The bakery filled the street with the smell of fresh bread.",
    "Thunder rumbled in the distance as clouds gathered.",
    "The train arrived at the station right on schedule.",
    "A rainbow appeared after the afternoon storm passed.",
    "The village square was decorated for the harvest festival.",
    "Fog rolled in from the coast and covered the hillside.",
    "The pianist played a soft melody in the concert hall.",
    "An eagle circled high above the canyon walls.",
    "The campfire crackled under a canopy of stars.",
    "Leaves drifted down the stream and over the waterfall.",
    "The town square fountain splashed quietly at midday.",
    "A bicycle leaned against the fence outside the shop.",
    "The sunset painted the sky in shades of orange.",
    "The morning dew glistened on the freshly cut grass.",
    "An old windmill turned slowly in the afternoon breeze.",
    "The cathedral bells chimed every hour on the hour.",
    "A sailboat glided across the calm lake surface.",
    "The orchard was heavy with ripe apples in autumn.",
    "The museum hall echoed with the footsteps of visitors.",
    "A lantern swung from the porch of the farmhouse.",
    "The waterfall roared as it plunged into the pool below.",
    "Butterflies danced among the wildflowers in the field.",
    "The cobblestone alley led to a hidden courtyard.",
    "A gentle snow began to fall over the sleeping village.",
    "The observatory dome opened to reveal the night sky.",
    "The footbridge swayed slightly as the hiker crossed.",
    "A warm fire burned in the hearth of the stone cottage.",
    "The tide pool was full of tiny crabs and sea anemones.",
]


def compute_mean_activations(model, n_sentences: int = 50) -> Dict[str, torch.Tensor]:
    """Compute mean activations across neutral sentences.

    Used for mean ablation: replacing target activations with the average
    activation from diverse, neutral text instead of zeroing them out.

    Args:
        model: HookedTransformer model.
        n_sentences: Number of neutral sentences to average over.

    Returns:
        Dict mapping hook names to mean activation tensors.
    """
    device = model.cfg.device
    n_layers = model.cfg.n_layers

    # Collect all relevant hook names — focus on MLP hooks
    # (attn hook_result isn't always in default cache; our top edges are MLP-to-MLP)
    hook_names = []
    for layer in range(n_layers):
        hook_names.append(_mlp_out_hook(layer))

    # Accumulate activations
    activation_sums = {}
    activation_counts = {}

    sentences = NEUTRAL_SENTENCES[:n_sentences]

    for sentence in tqdm(sentences, desc="Computing mean activations"):
        tokens = model.to_tokens(sentence, prepend_bos=True).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        for hook_name in hook_names:
            if hook_name not in cache.cache_dict:
                continue
            act = cache[hook_name]  # [1, seq_len, d_model]
            # Average over batch and sequence dimensions
            act_mean = act.mean(dim=(0, 1))  # [d_model]
            if hook_name not in activation_sums:
                activation_sums[hook_name] = act_mean.clone()
                activation_counts[hook_name] = 1
            else:
                activation_sums[hook_name] += act_mean
                activation_counts[hook_name] += 1

    # Compute means
    mean_activations = {}
    for hook_name in activation_sums:
        mean_activations[hook_name] = activation_sums[hook_name] / activation_counts[hook_name]

    print(f"[intervention] Computed mean activations from {len(sentences)} neutral sentences")
    print(f"[intervention] Cached {len(mean_activations)} hook points")
    return mean_activations


def _build_patch_hooks(
    model,
    edges: List[Edge],
    corrupted_cache,
    alpha: float = 1.0,
) -> list:
    """Build forward hooks that patch specified edges with blended activations.

    V2: Uses alpha blending instead of full replacement.
    new_activation = alpha * corrupted + (1 - alpha) * clean

    Args:
        model: HookedTransformer model.
        edges: List of Edge objects to patch.
        corrupted_cache: ActivationCache from the corrupted forward pass.
        alpha: Blending factor (0.0 = no change, 1.0 = full replacement).

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

            def make_attn_hook(hname, heads, a):
                def hook_fn(activation, hook):
                    corrupted_act = corrupted_cache[hname]
                    for head in heads:
                        activation[:, :, head, :] = (
                            a * corrupted_act[:, :, head, :] +
                            (1 - a) * activation[:, :, head, :]
                        )
                    return activation
                return hook_fn

            hooks.append((hook_name, make_attn_hook(hook_name, heads_to_patch, alpha)))

        else:  # MLP
            def make_mlp_hook(hname, a):
                def hook_fn(activation, hook):
                    corrupted_act = corrupted_cache[hname]
                    return a * corrupted_act + (1 - a) * activation
                return hook_fn

            hooks.append((hook_name, make_mlp_hook(hook_name, alpha)))

    return hooks


def build_mean_ablation_hooks(
    edges: List[Edge],
    mean_activations: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> list:
    """Build hooks that replace activations with precomputed mean values.

    Used for CoLA evaluation where corrupted counterparts don't exist.
    V2: Uses mean ablation instead of zero ablation.

    Args:
        edges: List of Edge objects to ablate.
        mean_activations: Dict from compute_mean_activations.
        alpha: Blending factor (0.0 = no change, 1.0 = full mean replacement).

    Returns:
        List of (hook_name, hook_fn) tuples.
    """
    dst_groups: Dict[str, dict] = {}
    for edge in edges:
        if edge.dst_type == "attn":
            hook_name = _attn_out_hook(edge.dst_layer)
        else:
            hook_name = _mlp_out_hook(edge.dst_layer)
        if hook_name not in dst_groups:
            dst_groups[hook_name] = {"type": edge.dst_type, "heads": set()}
        if edge.dst_head is not None:
            dst_groups[hook_name]["heads"].add(edge.dst_head)

    hooks = []
    for hook_name, info in dst_groups.items():
        mean_act = mean_activations.get(hook_name)
        if mean_act is None:
            continue

        if info["type"] == "attn" and info["heads"]:
            heads = list(info["heads"])
            def make_hook(h_list, m_act, a):
                def hook_fn(activation, hook):
                    for h in h_list:
                        # Expand mean to match activation shape
                        mean_expanded = m_act[h, :].unsqueeze(0).unsqueeze(0)
                        activation[:, :, h, :] = (
                            a * mean_expanded + (1 - a) * activation[:, :, h, :]
                        )
                    return activation
                return hook_fn
            hooks.append((hook_name, make_hook(heads, mean_act, alpha)))
        elif info["type"] == "mlp":
            def make_mlp_hook(m_act, a):
                def hook_fn(activation, hook):
                    mean_expanded = m_act.unsqueeze(0).unsqueeze(0)
                    return a * mean_expanded + (1 - a) * activation
                return hook_fn
            hooks.append((hook_name, make_mlp_hook(mean_act, alpha)))

    return hooks


def ablate_edges(
    model,
    edges: List[Edge],
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Run the model with specified edges patched (ablated).

    V2: Uses alpha blending for partial patching.

    Args:
        model: HookedTransformer model.
        edges: List of Edge objects to ablate.
        clean_tokens: Tokenized clean prompt [1, seq_len].
        corrupted_tokens: Tokenized corrupted prompt [1, seq_len].
        alpha: Blending factor (0 = no intervention, 1 = full replacement).

    Returns:
        Logits tensor [1, seq_len, vocab_size] from the patched forward pass.
    """
    device = model.cfg.device
    clean_tokens = clean_tokens.to(device)
    corrupted_tokens = corrupted_tokens.to(device)

    # Get corrupted activations
    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # Build patching hooks with alpha blending
    patch_hooks = _build_patch_hooks(model, edges, corrupted_cache, alpha=alpha)

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
    alpha: float = 1.0,
) -> Dict:
    """Compute bias score after intervention for all prompt pairs.

    Args:
        model: HookedTransformer model.
        edges: List of Edge objects to ablate.
        prompt_pairs: List of dicts from create_prompt_pairs.
        male_ids: Token IDs for male-gendered words.
        female_ids: Token IDs for female-gendered words.
        alpha: Blending factor for intervention.

    Returns:
        Dict with per-prompt results and aggregate statistics.
    """
    results = []
    min_len = min(len(male_ids), len(female_ids))

    for pair in tqdm(prompt_pairs, desc=f"Measuring debiased scores (alpha={alpha})"):
        logits = ablate_edges(
            model, edges,
            pair["clean_tokens"],
            pair["corrupted_tokens"],
            alpha=alpha,
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
        "alpha": alpha,
    }

    print(f"\n[intervention] Mean bias after: {summary['mean_bias_after']:.4f}")
    print(f"[intervention] Edges ablated: {len(edges)}, alpha: {alpha}")

    return summary


def run_debiasing_comparison(
    model,
    edges: List[Edge],
    prompt_pairs: List[Dict],
    male_ids: List[int],
    female_ids: List[int],
    baseline_results: Dict,
    alpha: float = 1.0,
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
        alpha: Blending factor for intervention.
        save_path: Optional path to save comparison results.

    Returns:
        Dict with before, after, and reduction statistics.
    """
    after = measure_debiased_score(
        model, edges, prompt_pairs, male_ids, female_ids, alpha=alpha
    )

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
        "alpha": alpha,
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
    print(f"  DEBIASING RESULTS (alpha={alpha})")
    print(f"{'=' * 60}")
    print(f"  Mean bias BEFORE: {before_mean:.4f}")
    print(f"  Mean bias AFTER:  {after_mean:.4f}")
    print(f"  Reduction:        {reduction:.1f}%")
    print(f"  Edges ablated:    {len(edges)}")
    print(f"  Alpha:            {alpha}")
    print(f"{'=' * 60}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"[intervention] Results saved to {save_path}")

    return comparison
