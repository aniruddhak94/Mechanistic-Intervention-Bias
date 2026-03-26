"""
baseline_scoring.py — Compute the L2 bias metric for language models.

Measures how differently the model predicts male vs. female gendered
continuations for a given prompt. Based on the L2-norm metric from the
Mechanistic Interpretability of Bias paper.
"""

import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_bias_score(
    model,
    clean_tokens: torch.Tensor,
    male_ids: List[int],
    female_ids: List[int],
) -> float:
    """Compute the L2 bias score for a single prompt.

    Bias = ||log_softmax(male_logits) - log_softmax(female_logits)||_2

    A higher score means the model more sharply distinguishes between
    male and female continuations — i.e., more bias.

    Args:
        model: HookedTransformer model.
        clean_tokens: Tokenized prompt, shape [1, seq_len].
        male_ids: List of token IDs for male-gendered words.
        female_ids: List of token IDs for female-gendered words.

    Returns:
        Bias score (float, ≥ 0).
    """
    with torch.no_grad():
        logits = model(clean_tokens)                       # [1, seq_len, vocab_size]
        last_logits = logits[0, -1, :]                     # [vocab_size]
        log_probs = F.log_softmax(last_logits, dim=-1)     # [vocab_size]

    male_log_probs = log_probs[male_ids]                   # [n_male]
    female_log_probs = log_probs[female_ids]               # [n_female]

    # Match lengths (take min of both lists)
    min_len = min(len(male_log_probs), len(female_log_probs))
    male_log_probs = male_log_probs[:min_len]
    female_log_probs = female_log_probs[:min_len]

    bias_score = torch.norm(male_log_probs - female_log_probs, p=2).item()
    return bias_score


def compute_directional_bias(
    model,
    clean_tokens: torch.Tensor,
    male_ids: List[int],
    female_ids: List[int],
) -> Dict:
    """Compute detailed directional bias information.

    Returns not just the L2 score but also which direction the bias leans.

    Args:
        model: HookedTransformer model.
        clean_tokens: Tokenized prompt, shape [1, seq_len].
        male_ids: Token IDs for male words.
        female_ids: Token IDs for female words.

    Returns:
        Dict with keys: bias_score, male_prob, female_prob, direction.
    """
    with torch.no_grad():
        logits = model(clean_tokens)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        log_probs = F.log_softmax(last_logits, dim=-1)

    male_prob = probs[male_ids].sum().item()
    female_prob = probs[female_ids].sum().item()

    min_len = min(len(male_ids), len(female_ids))
    bias_score = torch.norm(
        log_probs[male_ids[:min_len]] - log_probs[female_ids[:min_len]], p=2
    ).item()

    direction = "male" if male_prob > female_prob else "female"

    return {
        "bias_score": bias_score,
        "male_prob": male_prob,
        "female_prob": female_prob,
        "direction": direction,
    }


def run_baseline(
    model,
    dataset: List[Dict],
    male_ids: List[int],
    female_ids: List[int],
    save_path: str = None,
) -> Dict:
    """Compute bias scores for all prompts in a dataset.

    Args:
        model: HookedTransformer model.
        dataset: List of tokenized prompt-pair dicts (from create_prompt_pairs).
        male_ids: Token IDs for male words.
        female_ids: Token IDs for female words.
        save_path: Optional path to save results JSON.

    Returns:
        Dict with keys:
            - per_prompt: list of per-prompt results
            - mean_bias: average bias score
            - max_bias: maximum bias score
            - min_bias: minimum bias score
    """
    results = []

    for entry in tqdm(dataset, desc="Computing baseline bias"):
        clean_tokens = entry["clean_tokens"].to(model.cfg.device)
        detail = compute_directional_bias(model, clean_tokens, male_ids, female_ids)
        detail["id"] = entry["id"]
        detail["clean_prompt"] = entry["clean_prompt"]
        detail["corrupted_prompt"] = entry["corrupted_prompt"]
        results.append(detail)

    scores = [r["bias_score"] for r in results]
    summary = {
        "per_prompt": results,
        "mean_bias": sum(scores) / len(scores),
        "max_bias": max(scores),
        "min_bias": min(scores),
        "n_prompts": len(scores),
    }

    print(f"\n[baseline] Mean bias: {summary['mean_bias']:.4f}")
    print(f"[baseline] Max bias:  {summary['max_bias']:.4f}")
    print(f"[baseline] Min bias:  {summary['min_bias']:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[baseline] Results saved to {save_path}")

    return summary
