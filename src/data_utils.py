"""
data_utils.py — Dataset loading and prompt pair utilities.

Loads bias datasets from JSON, creates clean/corrupted token pairs,
and provides gendered token ID lists for bias scoring.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset


# ── Gendered word lists ──────────────────────────────────────────────────────

MALE_WORDS = [
    "he", "him", "his", "man", "men", "boy", "boys", "male", "father",
    "son", "husband", "brother", "gentleman", "sir", "king", "prince",
    "himself", "mr", "businessman", "uncle",
]

FEMALE_WORDS = [
    "she", "her", "hers", "woman", "women", "girl", "girls", "female",
    "mother", "daughter", "wife", "sister", "lady", "madam", "queen",
    "princess", "herself", "mrs", "businesswoman", "aunt",
]


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_bias_dataset(path: str) -> List[Dict]:
    """Load a bias dataset from a JSON file.

    Args:
        path: Path to JSON file containing list of prompt pair dicts.

    Returns:
        List of dicts with keys: id, clean_prompt, corrupted_prompt, etc.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[data_utils] Loaded {len(data)} prompt pairs from {os.path.basename(path)}")
    return data


def create_prompt_pairs(
    dataset: List[Dict],
    model,
) -> List[Dict]:
    """Tokenize clean and corrupted prompts and return paired token tensors.

    Args:
        dataset: List of dicts from load_bias_dataset.
        model: A TransformerLens HookedTransformer (has .to_tokens method).

    Returns:
        List of dicts with keys:
            - id: str
            - clean_tokens: Tensor [1, seq_len]
            - corrupted_tokens: Tensor [1, seq_len]
            - clean_prompt: str
            - corrupted_prompt: str
    """
    pairs = []
    for entry in dataset:
        clean_tok = model.to_tokens(entry["clean_prompt"], prepend_bos=True)
        corr_tok = model.to_tokens(entry["corrupted_prompt"], prepend_bos=True)

        # Pad or truncate to match sequence lengths
        max_len = max(clean_tok.shape[1], corr_tok.shape[1])
        if clean_tok.shape[1] < max_len:
            pad = torch.zeros(1, max_len - clean_tok.shape[1], dtype=clean_tok.dtype,
                              device=clean_tok.device)
            clean_tok = torch.cat([clean_tok, pad], dim=1)
        if corr_tok.shape[1] < max_len:
            pad = torch.zeros(1, max_len - corr_tok.shape[1], dtype=corr_tok.dtype,
                              device=corr_tok.device)
            corr_tok = torch.cat([corr_tok, pad], dim=1)

        pairs.append({
            "id": entry["id"],
            "clean_tokens": clean_tok,
            "corrupted_tokens": corr_tok,
            "clean_prompt": entry["clean_prompt"],
            "corrupted_prompt": entry["corrupted_prompt"],
        })

    print(f"[data_utils] Created {len(pairs)} tokenized prompt pairs")
    return pairs


def get_gendered_token_ids(model) -> Tuple[List[int], List[int]]:
    """Get token IDs for male and female gendered words.

    Args:
        model: A TransformerLens HookedTransformer.

    Returns:
        (male_ids, female_ids): Lists of token IDs.
    """
    male_ids = []
    female_ids = []

    for word in MALE_WORDS:
        tokens = model.to_tokens(f" {word}", prepend_bos=False).squeeze()
        if tokens.dim() == 0:
            male_ids.append(tokens.item())
        else:
            # Take the first sub-token
            male_ids.append(tokens[0].item())

    for word in FEMALE_WORDS:
        tokens = model.to_tokens(f" {word}", prepend_bos=False).squeeze()
        if tokens.dim() == 0:
            female_ids.append(tokens.item())
        else:
            female_ids.append(tokens[0].item())

    # Remove duplicates while preserving order
    male_ids = list(dict.fromkeys(male_ids))
    female_ids = list(dict.fromkeys(female_ids))

    print(f"[data_utils] Male token IDs: {len(male_ids)}, Female token IDs: {len(female_ids)}")
    return male_ids, female_ids


# ── Downstream evaluation datasets ──────────────────────────────────────────

def load_cola_dataset(split: str = "validation") -> List[Dict]:
    """Load the CoLA (Corpus of Linguistic Acceptability) dataset.

    Args:
        split: Which split to load ('train', 'validation', 'test').

    Returns:
        List of dicts with keys: sentence, label (1=acceptable, 0=unacceptable).
    """
    ds = load_dataset("glue", "cola", split=split)
    examples = []
    for item in ds:
        examples.append({
            "sentence": item["sentence"],
            "label": item["label"],
        })
    print(f"[data_utils] Loaded {len(examples)} CoLA examples ({split} split)")
    return examples
