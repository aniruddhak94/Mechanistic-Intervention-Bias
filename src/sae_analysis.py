"""
sae_analysis.py -- Sparse Autoencoder analysis for gender feature discovery.

Uses pre-trained SAEs (via SAELens) to decompose polysemantic MLP activations
into monosemantic features, then identifies gender-specific features by
measuring differential activation between male/female prompt pairs.

V3 Core Module.

References:
  - Cunningham et al., 2023: "Sparse Autoencoders Find Highly Interpretable
    Features in Language Models"
  - Templeton et al., 2024: "Scaling Monosemanticity" (Anthropic)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def load_pretrained_sae(layer: int, device: str = "cpu"):
    """Load a pre-trained SAE from HuggingFace for a specific GPT-2 layer.

    Uses SAELens to load residual-stream SAEs trained by Joseph Bloom.

    Args:
        layer: GPT-2 layer index (0-11).
        device: Device to load onto.

    Returns:
        Tuple of (sae, cfg_dict, sparsity).
    """
    from sae_lens import SAE

    release = "gpt2-small-res-jb"
    sae_id = f"blocks.{layer}.hook_resid_pre"

    print(f"[SAE] Loading pre-trained SAE for layer {layer} ({sae_id})...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    print(f"[SAE] Loaded: {sae_id}, d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    return sae, cfg_dict, sparsity


def find_gender_features(
    model,
    sae,
    prompt_pairs: List[Dict],
    male_ids: List[int],
    female_ids: List[int],
    layer: int,
    top_k: int = 50,
) -> List[Dict]:
    """Identify SAE features associated with gender bias.

    For each prompt pair, encodes the residual stream activations through
    the SAE and measures which features activate differently for
    male-stereotyped vs female-stereotyped prompts.

    Args:
        model: HookedTransformer model.
        sae: Pre-trained SAE for the target layer.
        prompt_pairs: Prompt pairs from create_prompt_pairs.
        male_ids: Male token IDs.
        female_ids: Female token IDs.
        layer: Layer index for the SAE.
        top_k: Number of top gender features to return.

    Returns:
        List of dicts with feature_idx, gender_score, direction, etc.
    """
    device = model.cfg.device
    hook_name = f"blocks.{layer}.hook_resid_pre"

    # Accumulate differential activations per feature
    n_features = sae.cfg.d_sae
    feature_diffs = torch.zeros(n_features, device=device)
    feature_male_means = torch.zeros(n_features, device=device)
    feature_female_means = torch.zeros(n_features, device=device)
    n_pairs = 0

    for pair in tqdm(prompt_pairs, desc=f"SAE feature scan (layer {layer})"):
        clean_tokens = pair["clean_tokens"].to(device)
        corrupted_tokens = pair["corrupted_tokens"].to(device)

        # Get residual stream activations at the target layer
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                clean_tokens,
                names_filter=[hook_name],
            )
            _, corrupted_cache = model.run_with_cache(
                corrupted_tokens,
                names_filter=[hook_name],
            )

        # Extract last-token activation
        clean_act = clean_cache[hook_name][0, -1, :]  # [d_model]
        corrupted_act = corrupted_cache[hook_name][0, -1, :]  # [d_model]

        # Encode through SAE
        clean_features = sae.encode(clean_act.unsqueeze(0))  # [1, d_sae]
        corrupted_features = sae.encode(corrupted_act.unsqueeze(0))  # [1, d_sae]

        # The clean prompt has the gendered occupation; corrupted is neutral.
        # Features that activate differently = features encoding gender info.
        diff = (clean_features - corrupted_features).abs().squeeze(0)
        feature_diffs += diff
        feature_male_means += clean_features.squeeze(0)
        feature_female_means += corrupted_features.squeeze(0)
        n_pairs += 1

    # Average
    feature_diffs /= n_pairs
    feature_male_means /= n_pairs
    feature_female_means /= n_pairs

    # Rank by gender relevance
    scores, indices = torch.sort(feature_diffs, descending=True)

    results = []
    for rank in range(min(top_k, n_features)):
        idx = indices[rank].item()
        score = scores[rank].item()
        m_act = feature_male_means[idx].item()
        f_act = feature_female_means[idx].item()
        direction = "male" if m_act > f_act else "female"

        results.append({
            "feature_idx": idx,
            "gender_score": round(score, 6),
            "male_mean_activation": round(m_act, 6),
            "female_mean_activation": round(f_act, 6),
            "direction": direction,
            "layer": layer,
            "rank": rank + 1,
        })

    return results


def discover_gender_features_multilayer(
    model,
    prompt_pairs: List[Dict],
    male_ids: List[int],
    female_ids: List[int],
    layers: List[int] = None,
    top_k_per_layer: int = 20,
) -> Dict:
    """Run SAE gender feature discovery across multiple layers.

    Args:
        model: HookedTransformer.
        prompt_pairs: Prompt pairs.
        male_ids: Male token IDs.
        female_ids: Female token IDs.
        layers: Layers to analyze (default: [6, 8, 9, 10, 11]).
        top_k_per_layer: Features per layer.

    Returns:
        Dict with per-layer features and aggregate statistics.
    """
    if layers is None:
        layers = [6, 8, 9, 10, 11]  # V2's top bias layers

    device = model.cfg.device
    all_features = {}
    layer_summaries = []

    for layer in layers:
        print(f"\n{'='*50}")
        print(f"  Analyzing Layer {layer}")
        print(f"{'='*50}")

        sae, cfg_dict, sparsity = load_pretrained_sae(layer, device=device)
        features = find_gender_features(
            model, sae, prompt_pairs, male_ids, female_ids,
            layer=layer, top_k=top_k_per_layer,
        )

        all_features[f"layer_{layer}"] = features

        # Summary stats
        top_score = features[0]["gender_score"] if features else 0.0
        mean_score = np.mean([f["gender_score"] for f in features[:10]])
        male_count = sum(1 for f in features if f["direction"] == "male")
        female_count = sum(1 for f in features if f["direction"] == "female")

        summary = {
            "layer": layer,
            "top_feature_score": round(top_score, 6),
            "mean_top10_score": round(float(mean_score), 6),
            "n_features_found": len(features),
            "male_biased_features": male_count,
            "female_biased_features": female_count,
            "d_sae": sae.cfg.d_sae,
        }
        layer_summaries.append(summary)

        print(f"  Top feature: idx={features[0]['feature_idx']}, "
              f"score={top_score:.4f}, dir={features[0]['direction']}")
        print(f"  Gender features: {male_count} male, {female_count} female")

        # Free memory
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {
        "features_per_layer": all_features,
        "layer_summaries": layer_summaries,
        "config": {
            "layers_analyzed": layers,
            "top_k_per_layer": top_k_per_layer,
            "n_prompt_pairs": len(prompt_pairs),
        },
    }

    return results


def build_feature_ablation_hooks(
    model,
    gender_features: Dict,
    layers: List[int],
    alpha: float = 1.0,
    method: str = "zero",
):
    """Build hooks that ablate specific SAE features during inference.

    This is the core V3 intervention: instead of ablating entire edges,
    we ablate only the gender-specific features identified by SAE analysis.

    Args:
        model: HookedTransformer.
        gender_features: Dict from discover_gender_features_multilayer.
        layers: Which layers to intervene on.
        alpha: Strength of ablation (0=none, 1=full).
        method: "zero" (set to 0) or "mean" (set to mean activation).

    Returns:
        List of (hook_name, hook_fn) tuples.
    """
    hooks = []

    for layer in layers:
        layer_key = f"layer_{layer}"
        if layer_key not in gender_features.get("features_per_layer", {}):
            continue

        feature_list = gender_features["features_per_layer"][layer_key]
        feature_indices = [f["feature_idx"] for f in feature_list]
        hook_name = f"blocks.{layer}.hook_resid_pre"

        # Load SAE for this layer
        device = model.cfg.device
        sae, _, _ = load_pretrained_sae(layer, device=device)

        # Compute target values for ablated features
        if method == "mean":
            mean_vals = torch.tensor(
                [f.get("female_mean_activation", 0.0)
                 if f["direction"] == "male"
                 else f.get("male_mean_activation", 0.0)
                 for f in feature_list],
                device=device,
            )
        else:
            mean_vals = None

        def make_hook(sae_obj, feat_indices, a, mean_targets):
            def hook_fn(activation, hook):
                # activation shape: [batch, seq_len, d_model]
                batch, seq_len, d_model = activation.shape
                act_flat = activation.reshape(-1, d_model)  # [B*S, d_model]

                # Encode through SAE
                features = sae_obj.encode(act_flat)  # [B*S, d_sae]

                # Ablate gender features
                for i, fidx in enumerate(feat_indices):
                    if mean_targets is not None:
                        target = mean_targets[i]
                    else:
                        target = 0.0
                    features[:, fidx] = (
                        a * target + (1 - a) * features[:, fidx]
                    )

                # Decode back
                reconstructed = sae_obj.decode(features)  # [B*S, d_model]
                reconstructed = reconstructed.reshape(batch, seq_len, d_model)

                # Blend with original
                return a * reconstructed + (1 - a) * activation

            return hook_fn

        hooks.append((
            hook_name,
            make_hook(sae, feature_indices, alpha, mean_vals),
        ))

    return hooks


def compute_feature_statistics(gender_features: Dict) -> Dict:
    """Compute aggregate statistics for thesis reporting.

    Args:
        gender_features: Results from discover_gender_features_multilayer.

    Returns:
        Dict with thesis-ready statistics.
    """
    all_scores = []
    all_features = []
    total_sae_features = 0

    for layer_key, features in gender_features.get("features_per_layer", {}).items():
        all_scores.extend([f["gender_score"] for f in features])
        all_features.extend(features)
        if features:
            # d_sae from layer summaries
            for s in gender_features.get("layer_summaries", []):
                if f"layer_{s['layer']}" == layer_key:
                    total_sae_features += s.get("d_sae", 0)
                    break

    if not all_scores:
        return {}

    scores_arr = np.array(all_scores)

    stats = {
        "total_gender_features_found": len(all_features),
        "total_sae_features_across_layers": total_sae_features,
        "gender_feature_ratio": round(
            len(all_features) / max(total_sae_features, 1) * 100, 4
        ),
        "mean_gender_score": round(float(scores_arr.mean()), 6),
        "std_gender_score": round(float(scores_arr.std()), 6),
        "max_gender_score": round(float(scores_arr.max()), 6),
        "min_gender_score": round(float(scores_arr.min()), 6),
        "male_biased_count": sum(1 for f in all_features if f["direction"] == "male"),
        "female_biased_count": sum(
            1 for f in all_features if f["direction"] == "female"
        ),
    }

    return stats
