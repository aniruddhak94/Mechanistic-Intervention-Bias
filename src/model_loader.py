"""
model_loader.py — Load GPT-2 via TransformerLens for mechanistic interpretability.

Provides a simple interface to load a HookedTransformer model with
automatic device detection (CUDA / CPU).
"""

import torch
from transformer_lens import HookedTransformer


def load_model(
    model_name: str = "gpt2",
    device: str = "auto",
) -> HookedTransformer:
    """Load a GPT-2 model using TransformerLens.

    Args:
        model_name: Name of the model to load. Options:
            - 'gpt2'        (124M params — recommended for free-tier GPUs)
            - 'gpt2-medium' (355M params — needs ~8GB GPU RAM)
            - 'gpt2-large'  (774M params — needs ~16GB GPU RAM)
        device: Device to load the model on.
            - 'auto': Use CUDA if available, else CPU.
            - 'cuda': Force CUDA.
            - 'cpu': Force CPU.

    Returns:
        HookedTransformer model ready for inference and hook-based analysis.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[model_loader] Loading {model_name} on {device}...")

    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
    )

    # Print model summary
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    print(f"[model_loader] [OK] Model loaded: {model_name}")
    print(f"  Layers: {n_layers}, Heads: {n_heads}, d_model: {d_model}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def get_model_info(model: HookedTransformer) -> dict:
    """Return a summary dict of model architecture details.

    Args:
        model: A loaded HookedTransformer.

    Returns:
        Dict with keys: model_name, n_layers, n_heads, d_model, d_head, d_mlp, device.
    """
    return {
        "model_name": model.cfg.model_name,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "d_head": model.cfg.d_head,
        "d_mlp": model.cfg.d_mlp,
        "device": str(model.cfg.device),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
