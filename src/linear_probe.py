"""
linear_probe.py -- Gender direction discovery via linear probing and CAA.

Implements two complementary techniques:
  1. Linear Probing: Train a logistic regression on MLP activations to find
     which layers encode gender most strongly. The probe's weight vector
     IS the "gender direction" in that layer's activation space.

  2. Contrastive Activation Addition (CAA): Compute a steering vector by
     averaging the difference in residual stream activations between
     male-stereotyped and female-stereotyped prompts.

  3. LEACE-style Projection: Compute a closed-form projection that removes
     gender information from activations with minimal damage.

V3 Module.

References:
  - Rimsky et al., 2024: "Steering Llama 2 via Contrastive Activation Addition"
  - Belrose et al., 2023: "LEACE: Perfect linear concept erasure in closed form"
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def collect_layer_activations(
    model,
    prompt_pairs: List[Dict],
    layers: List[int],
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Collect residual stream activations labeled by gender direction.

    For each prompt pair, we collect the last-token activation at each
    target layer from both the clean (gendered) and corrupted (neutral) prompt.

    Args:
        model: HookedTransformer.
        prompt_pairs: Prompt pairs with clean_tokens and corrupted_tokens.
        layers: List of layer indices.

    Returns:
        Dict mapping layer -> (activations_tensor [N, d_model], labels_tensor [N])
        Labels: 0 = from gendered prompt, 1 = from neutral prompt.
    """
    device = model.cfg.device
    hook_names = [f"blocks.{l}.hook_resid_pre" for l in layers]

    layer_acts = {l: ([], []) for l in layers}  # (activations, labels)

    for pair in tqdm(prompt_pairs, desc="Collecting activations"):
        clean_tok = pair["clean_tokens"].to(device)
        corrupted_tok = pair["corrupted_tokens"].to(device)

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                clean_tok, names_filter=hook_names
            )
            _, corrupted_cache = model.run_with_cache(
                corrupted_tok, names_filter=hook_names
            )

        for layer in layers:
            hname = f"blocks.{layer}.hook_resid_pre"
            # Last token activation
            c_act = clean_cache[hname][0, -1, :].cpu()
            n_act = corrupted_cache[hname][0, -1, :].cpu()
            layer_acts[layer][0].append(c_act)
            layer_acts[layer][0].append(n_act)
            layer_acts[layer][1].append(0)  # gendered
            layer_acts[layer][1].append(1)  # neutral

    # Stack into tensors
    result = {}
    for layer in layers:
        acts = torch.stack(layer_acts[layer][0])
        labels = torch.tensor(layer_acts[layer][1])
        result[layer] = (acts, labels)

    return result


def train_gender_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    test_split: float = 0.2,
) -> Dict:
    """Train a linear probe (logistic regression) on gender-labeled activations.

    Args:
        activations: Tensor [N, d_model].
        labels: Tensor [N] — 0=gendered, 1=neutral.
        test_split: Fraction for test set.

    Returns:
        Dict with accuracy, weight_vector, bias, and per-class metrics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    X = activations.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")

    # The weight vector is the "gender direction"
    weight_vector = torch.tensor(clf.coef_[0], dtype=torch.float32)
    weight_vector = weight_vector / weight_vector.norm()  # Normalize

    return {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "gender_direction": weight_vector,
        "bias": float(clf.intercept_[0]),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def run_probing_all_layers(
    model,
    prompt_pairs: List[Dict],
    layers: List[int] = None,
) -> Dict:
    """Run linear probing across all specified layers.

    Args:
        model: HookedTransformer.
        prompt_pairs: Prompt pairs.
        layers: Layers to probe (default: 0-11).

    Returns:
        Dict with per-layer results and the overall best layer.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    print("\n[Probe] Collecting activations across all layers...")
    layer_data = collect_layer_activations(model, prompt_pairs, layers)

    results = {}
    best_layer = -1
    best_accuracy = 0.0

    for layer in layers:
        acts, labels = layer_data[layer]
        print(f"[Probe] Training probe for layer {layer}...", end=" ")
        probe_result = train_gender_probe(acts, labels)
        print(f"accuracy={probe_result['accuracy']:.3f}, "
              f"F1={probe_result['f1_score']:.3f}")

        # Store without the weight vector (too large for JSON)
        result_for_json = {k: v for k, v in probe_result.items()
                          if k != "gender_direction"}
        result_for_json["layer"] = layer
        results[f"layer_{layer}"] = result_for_json

        if probe_result["accuracy"] > best_accuracy:
            best_accuracy = probe_result["accuracy"]
            best_layer = layer

    return {
        "per_layer": results,
        "best_layer": best_layer,
        "best_accuracy": round(best_accuracy, 4),
        "n_layers_probed": len(layers),
    }


def compute_caa_steering_vectors(
    model,
    prompt_pairs: List[Dict],
    layers: List[int] = None,
) -> Dict[int, torch.Tensor]:
    """Compute Contrastive Activation Addition (CAA) steering vectors.

    The steering vector at each layer is the mean difference between
    gendered and neutral prompt activations.

    Args:
        model: HookedTransformer.
        prompt_pairs: Prompt pairs.
        layers: Target layers (default: [6, 8, 9, 10, 11]).

    Returns:
        Dict mapping layer -> steering_vector (torch.Tensor [d_model]).
    """
    if layers is None:
        layers = [6, 8, 9, 10, 11]

    print("\n[CAA] Computing steering vectors...")
    layer_data = collect_layer_activations(model, prompt_pairs, layers)

    steering_vectors = {}
    for layer in layers:
        acts, labels = layer_data[layer]
        gendered_acts = acts[labels == 0]  # Clean/gendered prompts
        neutral_acts = acts[labels == 1]   # Corrupted/neutral prompts

        # Steering vector = mean(gendered) - mean(neutral)
        steering = gendered_acts.mean(dim=0) - neutral_acts.mean(dim=0)
        # Normalize to unit vector
        steering = steering / steering.norm()

        steering_vectors[layer] = steering
        print(f"  Layer {layer}: ||steering|| = {steering.norm():.4f} (normalized)")

    return steering_vectors


def build_caa_hooks(
    steering_vectors: Dict[int, torch.Tensor],
    strength: float = 1.0,
    device: str = "cpu",
) -> list:
    """Build hooks that subtract the gender steering vector during inference.

    Args:
        steering_vectors: Dict from compute_caa_steering_vectors.
        strength: Multiplier for the steering vector.
        device: Target device.

    Returns:
        List of (hook_name, hook_fn) tuples.
    """
    hooks = []
    for layer, vec in steering_vectors.items():
        hook_name = f"blocks.{layer}.hook_resid_pre"
        steering = vec.to(device)

        def make_hook(sv, s):
            def hook_fn(activation, hook):
                # Subtract gender direction from all positions
                return activation - s * sv.unsqueeze(0).unsqueeze(0)
            return hook_fn

        hooks.append((hook_name, make_hook(steering, strength)))

    return hooks


def compute_leace_projection(
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute LEACE projection matrix for concept erasure.

    LEACE (LEAst-squares Concept Erasure) finds the projection that
    removes all linearly-extractable information about a concept
    with minimal distortion to the representation.

    Simplified binary LEACE: project onto the orthogonal complement
    of the gender direction (found via the class-conditional mean
    difference).

    Args:
        activations: Tensor [N, d_model].
        labels: Binary labels [N].

    Returns:
        Projection matrix [d_model, d_model].
    """
    d = activations.shape[1]

    # Class-conditional means
    mask_0 = labels == 0
    mask_1 = labels == 1
    mu_0 = activations[mask_0].mean(dim=0)
    mu_1 = activations[mask_1].mean(dim=0)

    # Gender direction = normalized difference of means
    delta = mu_0 - mu_1
    delta = delta / delta.norm()

    # LEACE projection: P = I - delta * delta^T
    # This projects out the gender direction
    P = torch.eye(d) - torch.outer(delta, delta)

    return P


def build_leace_hooks(
    model,
    prompt_pairs: List[Dict],
    layers: List[int] = None,
    alpha: float = 1.0,
) -> list:
    """Build hooks that apply LEACE projection during inference.

    Args:
        model: HookedTransformer.
        prompt_pairs: Prompt pairs.
        layers: Target layers.
        alpha: Blending factor (0=no erasure, 1=full erasure).

    Returns:
        List of (hook_name, hook_fn) tuples.
    """
    if layers is None:
        layers = [6, 8, 9, 10, 11]

    device = model.cfg.device
    layer_data = collect_layer_activations(model, prompt_pairs, layers)

    hooks = []
    for layer in layers:
        acts, labels = layer_data[layer]
        P = compute_leace_projection(acts, labels).to(device)
        hook_name = f"blocks.{layer}.hook_resid_pre"

        def make_hook(proj, a):
            def hook_fn(activation, hook):
                batch, seq_len, d_model = activation.shape
                flat = activation.reshape(-1, d_model)
                projected = flat @ proj.T
                projected = projected.reshape(batch, seq_len, d_model)
                return a * projected + (1 - a) * activation
            return hook_fn

        hooks.append((hook_name, make_hook(P, alpha)))
        print(f"[LEACE] Projection matrix ready for layer {layer}")

    return hooks
