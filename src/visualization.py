"""
visualization.py -- Plotting functions for bias intervention results.

Generates publication-quality figures for:
  1. Bias comparison bar chart (before/after per prompt)
  2. Edge circuit heatmap (layer-to-layer EAP scores)
  3. Perplexity comparison chart
  4. EAP score distribution histogram
  5. Bias reduction summary dashboard
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# -- Style configuration -------------------------------------------------------

COLORS = {
    "before": "#E74C3C",      # Red
    "after": "#2ECC71",       # Green
    "improved": "#27AE60",    # Dark green
    "worsened": "#C0392B",    # Dark red
    "neutral": "#95A5A6",     # Gray
    "primary": "#3498DB",     # Blue
    "secondary": "#9B59B6",   # Purple
    "accent": "#F39C12",      # Orange
    "bg": "#FAFAFA",          # Light background
    "grid": "#ECF0F1",        # Grid lines
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": COLORS["bg"],
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": COLORS["grid"],
})


def plot_bias_comparison(debiasing_results: Dict, output_path: str):
    """Horizontal bar chart: per-prompt bias before vs after.

    Shows each prompt with two bars (before/after), color-coded
    by whether the prompt improved or worsened.
    """
    per_prompt = debiasing_results["per_prompt"]

    # Sort by reduction (best improvements first)
    per_prompt_sorted = sorted(per_prompt, key=lambda x: x["reduction"], reverse=True)

    # Take top 30 for readability
    display = per_prompt_sorted[:30]

    labels = []
    for item in display:
        prompt = item["clean_prompt"]
        # Extract just the occupation
        if prompt.startswith("The "):
            words = prompt[4:].split()
            # Find words before the verb
            occ = []
            for w in words:
                if w[0].islower() and w not in ("the", "a", "an", "of", "in", "for", "and"):
                    break
                occ.append(w)
            label = " ".join(occ) if occ else words[0]
        else:
            label = prompt[:25]
        labels.append(label[:22])

    before_scores = [item["bias_before"] for item in display]
    after_scores = [item["bias_after"] for item in display]
    reductions = [item["reduction"] for item in display]

    y_pos = np.arange(len(labels))
    bar_height = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(8, len(labels) * 0.35)),
                                     gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: Before/After bars
    ax1.barh(y_pos + bar_height / 2, before_scores, bar_height,
             label="Before", color=COLORS["before"], alpha=0.8)
    ax1.barh(y_pos - bar_height / 2, after_scores, bar_height,
             label="After", color=COLORS["after"], alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("L2 Bias Score")
    ax1.set_title("Per-Prompt Bias: Before vs After Intervention")
    ax1.legend(loc="lower right")

    # Right panel: Reduction arrows
    colors = [COLORS["improved"] if r > 0 else COLORS["worsened"] for r in reductions]
    ax2.barh(y_pos, reductions, 0.6, color=colors, alpha=0.8)
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Bias Reduction")
    ax2.set_title("Change (+ = improved)")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved bias comparison to {output_path}")


def plot_edge_heatmap(edges_data: List[Dict], n_layers: int, output_path: str):
    """Layer-to-layer heatmap showing sum of EAP scores.

    Visualizes which layer pairs have the strongest bias-carrying connections.
    """
    # Build the heatmap matrix
    heatmap = np.zeros((n_layers, n_layers))
    for edge in edges_data:
        src = edge["src_layer"]
        dst = edge["dst_layer"]
        heatmap[src][dst] += edge["score"]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto",
                   interpolation="nearest")

    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Destination Layer")
    ax.set_ylabel("Source Layer")
    ax.set_title("EAP Score Heatmap: Bias Circuit Connectivity")

    # Add text annotations
    for i in range(n_layers):
        for j in range(n_layers):
            val = heatmap[i][j]
            if val > 0:
                text_color = "white" if val > heatmap.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label="Cumulative EAP Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved edge heatmap to {output_path}")


def plot_perplexity_comparison(cola_results: Dict, output_path: str,
                                v1_results: Optional[Dict] = None):
    """Bar chart comparing baseline vs ablated perplexity.

    If V1 results are provided, shows three bars: Baseline, V1 (zero), V2 (mean).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Baseline"]
    values = [cola_results["baseline_perplexity"]]
    colors = [COLORS["primary"]]

    if v1_results and "ablated_perplexity" in v1_results:
        categories.append(f"V1: Zero Ablation\n({v1_results.get('n_edges_ablated', '?')} edges)")
        values.append(min(v1_results["ablated_perplexity"], 2000))  # Cap for display
        colors.append(COLORS["before"])

    if "ablated_perplexity" in cola_results:
        alpha = cola_results.get("alpha", "?")
        categories.append(f"V2: Mean Ablation\n(alpha={alpha}, {cola_results.get('n_edges_ablated', '?')} edges)")
        values.append(cola_results["ablated_perplexity"])
        colors.append(COLORS["after"])

    bars = ax.bar(categories, values, color=colors, alpha=0.85, width=0.5,
                  edgecolor="white", linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        display_val = val
        label = f"{display_val:.1f}"
        if v1_results and val == min(v1_results.get("ablated_perplexity", 0), 2000):
            label = f"{v1_results['ablated_perplexity']:.0f}\n(capped)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("Collateral Damage: Perplexity Before and After Intervention")
    ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved perplexity comparison to {output_path}")


def plot_eap_distribution(edges_data: List[Dict], output_path: str):
    """Histogram and CDF of EAP scores showing the score distribution."""
    scores = [e["score"] for e in edges_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(scores, bins=25, color=COLORS["primary"], alpha=0.7,
             edgecolor="white", linewidth=0.8)
    ax1.axvline(x=np.mean(scores), color=COLORS["accent"], linestyle="--",
                linewidth=2, label=f"Mean: {np.mean(scores):.4f}")
    ax1.axvline(x=np.median(scores), color=COLORS["secondary"], linestyle="--",
                linewidth=2, label=f"Median: {np.median(scores):.4f}")
    ax1.set_xlabel("EAP Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of EAP Edge Scores")
    ax1.legend()

    # CDF
    sorted_scores = np.sort(scores)[::-1]
    cumulative = np.cumsum(sorted_scores) / np.sum(sorted_scores) * 100
    ax2.plot(range(1, len(sorted_scores) + 1), cumulative,
             color=COLORS["primary"], linewidth=2)
    ax2.axhline(y=80, color=COLORS["accent"], linestyle="--", alpha=0.5,
                label="80% of total score")
    ax2.fill_between(range(1, len(sorted_scores) + 1), cumulative,
                     alpha=0.1, color=COLORS["primary"])
    ax2.set_xlabel("Number of Top Edges")
    ax2.set_ylabel("Cumulative Score (%)")
    ax2.set_title("Cumulative EAP Score (Top Edges)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved EAP distribution to {output_path}")


def plot_bias_reduction_summary(debiasing_results: Dict, cola_results: Dict,
                                 output_path: str):
    """Summary dashboard with key metrics and a box plot."""
    per_prompt = debiasing_results["per_prompt"]
    reductions = [p["reduction"] for p in per_prompt]
    improved = sum(1 for r in reductions if r > 0)
    worsened = sum(1 for r in reductions if r < 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Box plot of reductions
    ax1 = axes[0]
    bp = ax1.boxplot(reductions, vert=True, patch_artist=True,
                     boxprops=dict(facecolor=COLORS["primary"], alpha=0.6),
                     medianprops=dict(color=COLORS["accent"], linewidth=2),
                     whiskerprops=dict(color=COLORS["primary"]),
                     capprops=dict(color=COLORS["primary"]))
    ax1.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax1.set_ylabel("Bias Reduction per Prompt")
    ax1.set_title("Distribution of Bias Changes")
    ax1.set_xticklabels(["All Prompts"])

    # Panel 2: Pie chart of improved vs worsened
    ax2 = axes[1]
    sizes = [improved, worsened]
    labels_pie = [f"Improved\n({improved})", f"Worsened\n({worsened})"]
    colors_pie = [COLORS["improved"], COLORS["worsened"]]
    ax2.pie(sizes, labels=labels_pie, colors=colors_pie, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 11})
    ax2.set_title("Prompts: Improved vs Worsened")

    # Panel 3: Key metrics text
    ax3 = axes[2]
    ax3.axis("off")

    before_mean = debiasing_results["before"]["mean_bias"]
    after_mean = debiasing_results["after"]["mean_bias"]
    reduction_pct = debiasing_results["reduction_percent"]
    n_edges = debiasing_results["n_edges_ablated"]
    alpha = debiasing_results.get("alpha", "N/A")

    ppl_baseline = cola_results.get("baseline_perplexity", "N/A")
    ppl_ablated = cola_results.get("ablated_perplexity", "N/A")
    ppl_increase = cola_results.get("perplexity_increase_percent", "N/A")

    metrics_text = (
        f"DEBIASING SUMMARY\n"
        f"{'=' * 35}\n\n"
        f"Mean Bias Before:  {before_mean:.4f}\n"
        f"Mean Bias After:   {after_mean:.4f}\n"
        f"Reduction:         {reduction_pct:.1f}%\n\n"
        f"Edges Ablated:     {n_edges}\n"
        f"Alpha:             {alpha}\n\n"
        f"{'=' * 35}\n"
        f"COLLATERAL DAMAGE\n"
        f"{'=' * 35}\n\n"
        f"Baseline PPL:      {ppl_baseline:.2f}\n"
        f"Ablated PPL:       {ppl_ablated:.2f}\n"
        f"PPL Increase:      {ppl_increase:.1f}%\n"
    )

    ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["bg"],
                      edgecolor=COLORS["grid"]))

    plt.suptitle("Mechanistic Bias Intervention -- Results Summary (V2)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved summary dashboard to {output_path}")
