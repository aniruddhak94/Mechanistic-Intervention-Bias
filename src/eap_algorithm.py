"""
eap_algorithm.py — Edge Attribution Patching for bias circuit discovery.

This module implements the core EAP algorithm from the Mechanistic
Interpretability of Bias paper.  EAP scores each edge (connection between
model components across layers) by how much it contributes to a given
bias metric.  High-scoring edges form the "bias circuit".

Algorithm (per prompt pair):
    1. Run the model on the CORRUPTED prompt, cache all intermediate
       activations (attention head outputs and MLP outputs).
    2. Run the model on the CLEAN prompt WITH gradient tracking.
       At each hook point, compute:
           activation_diff = clean_activation − corrupted_activation
    3. Back-propagate from the bias-metric scalar through the clean run.
    4. For each edge (src → dst), the EAP score is:
           score = (grad at dst w.r.t. input from src) · (activation_diff at src)
       Approximated as: grad_dst · act_diff_src  (summed over token and hidden dims).
    5. Aggregate across all prompt pairs and rank.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── Edge representation ──────────────────────────────────────────────────────

@dataclass
class Edge:
    """Represents a connection between two model components."""
    src_layer: int
    src_type: str     # 'attn' or 'mlp'
    src_head: Optional[int]   # None if MLP
    dst_layer: int
    dst_type: str     # 'attn' or 'mlp'
    dst_head: Optional[int]   # None if MLP
    score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "src_layer": self.src_layer,
            "src_type": self.src_type,
            "src_head": self.src_head,
            "dst_layer": self.dst_layer,
            "dst_type": self.dst_type,
            "dst_head": self.dst_head,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Edge":
        return cls(**d)

    def __repr__(self):
        src = f"L{self.src_layer}.{self.src_type}"
        if self.src_head is not None:
            src += f".H{self.src_head}"
        dst = f"L{self.dst_layer}.{self.dst_type}"
        if self.dst_head is not None:
            dst += f".H{self.dst_head}"
        return f"Edge({src} → {dst}, score={self.score:.6f})"


# ── Hook-point name helpers ──────────────────────────────────────────────────

def _attn_out_hook(layer: int) -> str:
    """Hook name for attention output at a given layer."""
    return f"blocks.{layer}.attn.hook_result"


def _mlp_out_hook(layer: int) -> str:
    """Hook name for MLP output at a given layer."""
    return f"blocks.{layer}.hook_mlp_out"


def _resid_pre_hook(layer: int) -> str:
    """Hook name for residual stream input to a given layer."""
    return f"blocks.{layer}.hook_resid_pre"


# ── Core EAP computation ────────────────────────────────────────────────────

def compute_eap_scores(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    male_ids: List[int],
    female_ids: List[int],
) -> List[Edge]:
    """Compute Edge Attribution Patching scores for one prompt pair.

    Args:
        model: HookedTransformer model.
        clean_tokens: Tokenized clean prompt [1, seq_len].
        corrupted_tokens: Tokenized corrupted prompt [1, seq_len].
        male_ids: Token IDs for male-gendered words.
        female_ids: Token IDs for female-gendered words.

    Returns:
        List of Edge objects with their EAP scores.
    """
    device = model.cfg.device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_tokens = clean_tokens.to(device)
    corrupted_tokens = corrupted_tokens.to(device)

    # ── Step 1: Corrupted forward pass (cache activations) ───────────────
    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # ── Step 2: Clean forward pass (with grad tracking) ──────────────────
    # We need gradients w.r.t. intermediate activations, so we use hooks
    # to store both the activations and enable gradient computation.

    clean_activations = {}
    clean_grads = {}

    def make_fwd_hook(name):
        """Create a forward hook that stores activations and enables grad."""
        def hook_fn(activation, hook):
            activation.retain_grad()
            clean_activations[name] = activation
            return activation
        return hook_fn

    # Register forward hooks for all relevant components
    hook_names = []
    for layer in range(n_layers):
        hook_names.append(_attn_out_hook(layer))
        hook_names.append(_mlp_out_hook(layer))

    fwd_hooks = [(name, make_fwd_hook(name)) for name in hook_names]

    # Run clean forward pass with hooks
    model.zero_grad()
    logits = model.run_with_hooks(
        clean_tokens,
        fwd_hooks=fwd_hooks,
    )

    # ── Step 3: Compute bias metric and backprop ─────────────────────────
    last_logits = logits[0, -1, :]
    log_probs = F.log_softmax(last_logits, dim=-1)

    min_len = min(len(male_ids), len(female_ids))
    male_lp = log_probs[male_ids[:min_len]]
    female_lp = log_probs[female_ids[:min_len]]

    # L2 bias metric (scalar)
    bias_metric = torch.norm(male_lp - female_lp, p=2)
    bias_metric.backward()

    # ── Step 4: Compute EAP scores per edge ──────────────────────────────
    edges = []

    for src_layer in range(n_layers):
        for dst_layer in range(src_layer + 1, n_layers):
            # Source: attention heads
            src_attn_hook = _attn_out_hook(src_layer)
            if src_attn_hook in clean_activations:
                src_act_clean = clean_activations[src_attn_hook]
                src_act_corr = corrupted_cache[src_attn_hook]
                act_diff = src_act_clean - src_act_corr  # [1, seq, n_heads, d_head]

                for src_head in range(n_heads):
                    head_diff = act_diff[0, :, src_head, :]  # [seq, d_head]

                    # Destination: attention heads in dst_layer
                    dst_attn_hook = _attn_out_hook(dst_layer)
                    if dst_attn_hook in clean_activations and clean_activations[dst_attn_hook].grad is not None:
                        dst_grad = clean_activations[dst_attn_hook].grad
                        for dst_head in range(n_heads):
                            grad_slice = dst_grad[0, :, dst_head, :]  # [seq, d_head]
                            # EAP score: sum over seq and hidden dims
                            score = (grad_slice * head_diff).sum().item()
                            edges.append(Edge(
                                src_layer=src_layer, src_type="attn", src_head=src_head,
                                dst_layer=dst_layer, dst_type="attn", dst_head=dst_head,
                                score=abs(score),
                            ))

                    # Destination: MLP in dst_layer
                    dst_mlp_hook = _mlp_out_hook(dst_layer)
                    if dst_mlp_hook in clean_activations and clean_activations[dst_mlp_hook].grad is not None:
                        dst_grad = clean_activations[dst_mlp_hook].grad  # [1, seq, d_model]
                        # Project head_diff into d_model space (sum over head dim)
                        head_diff_proj = head_diff.sum(dim=-1)  # [seq]
                        grad_sum = dst_grad[0].sum(dim=-1)      # [seq]
                        score = (grad_sum * head_diff_proj).sum().item()
                        edges.append(Edge(
                            src_layer=src_layer, src_type="attn", src_head=src_head,
                            dst_layer=dst_layer, dst_type="mlp", dst_head=None,
                            score=abs(score),
                        ))

            # Source: MLP
            src_mlp_hook = _mlp_out_hook(src_layer)
            if src_mlp_hook in clean_activations:
                src_act_clean = clean_activations[src_mlp_hook]   # [1, seq, d_model]
                src_act_corr = corrupted_cache[src_mlp_hook]
                act_diff = src_act_clean - src_act_corr           # [1, seq, d_model]

                # Destination: attention heads in dst_layer
                dst_attn_hook = _attn_out_hook(dst_layer)
                if dst_attn_hook in clean_activations and clean_activations[dst_attn_hook].grad is not None:
                    dst_grad = clean_activations[dst_attn_hook].grad
                    for dst_head in range(n_heads):
                        grad_slice = dst_grad[0, :, dst_head, :]  # [seq, d_head]
                        act_diff_proj = act_diff[0].sum(dim=-1)   # [seq]
                        grad_proj = grad_slice.sum(dim=-1)        # [seq]
                        score = (grad_proj * act_diff_proj).sum().item()
                        edges.append(Edge(
                            src_layer=src_layer, src_type="mlp", src_head=None,
                            dst_layer=dst_layer, dst_type="attn", dst_head=dst_head,
                            score=abs(score),
                        ))

                # Destination: MLP in dst_layer
                dst_mlp_hook = _mlp_out_hook(dst_layer)
                if dst_mlp_hook in clean_activations and clean_activations[dst_mlp_hook].grad is not None:
                    dst_grad = clean_activations[dst_mlp_hook].grad  # [1, seq, d_model]
                    score = (dst_grad[0] * act_diff[0]).sum().item()
                    edges.append(Edge(
                        src_layer=src_layer, src_type="mlp", src_head=None,
                        dst_layer=dst_layer, dst_type="mlp", dst_head=None,
                        score=abs(score),
                    ))

    # Clean up
    model.zero_grad()

    return edges


def aggregate_eap_scores(
    model,
    prompt_pairs: List[Dict],
    male_ids: List[int],
    female_ids: List[int],
) -> List[Edge]:
    """Run EAP across all prompt pairs and aggregate edge scores.

    Scores are averaged across all prompt pairs.

    Args:
        model: HookedTransformer model.
        prompt_pairs: List of dicts from create_prompt_pairs.
        male_ids: Token IDs for male-gendered words.
        female_ids: Token IDs for female-gendered words.

    Returns:
        List of Edge objects sorted by score (descending).
    """
    edge_totals: Dict[str, Edge] = {}

    for pair in tqdm(prompt_pairs, desc="Running EAP"):
        edges = compute_eap_scores(
            model,
            pair["clean_tokens"],
            pair["corrupted_tokens"],
            male_ids,
            female_ids,
        )

        for edge in edges:
            key = f"{edge.src_layer}_{edge.src_type}_{edge.src_head}_{edge.dst_layer}_{edge.dst_type}_{edge.dst_head}"
            if key in edge_totals:
                edge_totals[key].score += edge.score
            else:
                edge_totals[key] = Edge(
                    src_layer=edge.src_layer,
                    src_type=edge.src_type,
                    src_head=edge.src_head,
                    dst_layer=edge.dst_layer,
                    dst_type=edge.dst_type,
                    dst_head=edge.dst_head,
                    score=edge.score,
                )

    # Average scores
    n_pairs = len(prompt_pairs)
    for edge in edge_totals.values():
        edge.score /= n_pairs

    # Sort by score (descending)
    sorted_edges = sorted(edge_totals.values(), key=lambda e: e.score, reverse=True)

    print(f"\n[EAP] Total unique edges: {len(sorted_edges)}")
    print(f"[EAP] Top-5 edges:")
    for i, edge in enumerate(sorted_edges[:5]):
        print(f"  {i + 1}. {edge}")

    return sorted_edges


def get_top_edges(edges: List[Edge], top_k: int = 50) -> List[Edge]:
    """Extract the top-K highest-scoring edges.

    Args:
        edges: Sorted list of edges from aggregate_eap_scores.
        top_k: Number of top edges to return.

    Returns:
        List of top-K Edge objects.
    """
    top = edges[:top_k]
    print(f"[EAP] Selected top {len(top)} edges (out of {len(edges)})")
    return top


def save_edges(edges: List[Edge], path: str):
    """Save edges to a JSON file.

    Args:
        edges: List of Edge objects.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [e.to_dict() for e in edges]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[EAP] Saved {len(edges)} edges to {path}")


def load_edges(path: str) -> List[Edge]:
    """Load edges from a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of Edge objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    edges = [Edge.from_dict(d) for d in data]
    print(f"[EAP] Loaded {len(edges)} edges from {path}")
    return edges
