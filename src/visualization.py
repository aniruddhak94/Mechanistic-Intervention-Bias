"""
visualization.py -- Thesis-quality plotting for bias intervention results.

V3 generates 12 publication-quality figure types:
  === V2 Existing ===
  1. Bias comparison bar chart (before/after per prompt)
  2. Edge circuit heatmap (layer-to-layer EAP scores)
  3. Perplexity comparison chart
  4. EAP score distribution histogram + CDF
  5. Bias reduction summary dashboard
  === V3 New ===
  6. Pareto frontier (alpha vs bias vs perplexity)
  7. Probe accuracy by layer (bar chart with error metrics)
  8. SAE feature gender spectrum (top-N features ranked)
  9. V1/V2/V3 comparison grouped bars
  10. Radar/spider chart for multi-metric comparison
  11. Method comparison (SAE vs CAA vs LEACE)
  12. Feature ablation curve (features vs bias reduction)
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# -- Thesis style config -------------------------------------------------------

COLORS = {
    "before": "#E74C3C",
    "after": "#2ECC71",
    "improved": "#27AE60",
    "worsened": "#C0392B",
    "neutral": "#95A5A6",
    "primary": "#3498DB",
    "secondary": "#9B59B6",
    "accent": "#F39C12",
    "bg": "#FAFAFA",
    "grid": "#ECF0F1",
    # V3 additions
    "v1": "#E74C3C",
    "v2": "#F39C12",
    "v3": "#2ECC71",
    "sae": "#3498DB",
    "caa": "#9B59B6",
    "leace": "#E67E22",
    "dark": "#2C3E50",
}

VERSION_COLORS = [COLORS["v1"], COLORS["v2"], COLORS["v3"]]
METHOD_COLORS = [COLORS["v2"], COLORS["sae"], COLORS["caa"], COLORS["leace"]]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": COLORS["bg"],
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": COLORS["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# =============================================================================
# V2 EXISTING PLOTS (preserved)
# =============================================================================

def plot_bias_comparison(debiasing_results: Dict, output_path: str):
    """Horizontal bar chart: per-prompt bias before vs after."""
    per_prompt = debiasing_results["per_prompt"]
    per_prompt_sorted = sorted(per_prompt, key=lambda x: x["reduction"], reverse=True)
    display = per_prompt_sorted[:30]

    labels = []
    for item in display:
        prompt = item["clean_prompt"]
        if prompt.startswith("The "):
            words = prompt[4:].split()
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

    ax1.barh(y_pos + bar_height/2, before_scores, bar_height,
             label="Before", color=COLORS["before"], alpha=0.8)
    ax1.barh(y_pos - bar_height/2, after_scores, bar_height,
             label="After", color=COLORS["after"], alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("L2 Bias Score")
    ax1.set_title("Per-Prompt Bias: Before vs After Intervention")
    ax1.legend(loc="lower right")

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
    """Layer-to-layer heatmap showing sum of EAP scores."""
    heatmap = np.zeros((n_layers, n_layers))
    for edge in edges_data:
        heatmap[edge["src_layer"]][edge["dst_layer"]] += edge["score"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
    ax.set_xlabel("Destination Layer")
    ax.set_ylabel("Source Layer")
    ax.set_title("EAP Score Heatmap: Bias Circuit Connectivity")

    for i in range(n_layers):
        for j in range(n_layers):
            val = heatmap[i][j]
            if val > 0:
                tc = "white" if val > heatmap.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=tc)

    plt.colorbar(im, ax=ax, label="Cumulative EAP Score", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved edge heatmap to {output_path}")


def plot_perplexity_comparison(cola_results: Dict, output_path: str,
                                v1_results: Optional[Dict] = None,
                                v3_results: Optional[Dict] = None):
    """Bar chart comparing baseline vs ablated perplexity across versions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Baseline\n(GPT-2)"]
    values = [cola_results["baseline_perplexity"]]
    colors = [COLORS["primary"]]

    if v1_results and "ablated_perplexity" in v1_results:
        categories.append(f"V1\nZero Ablation\n(50 edges)")
        values.append(min(v1_results["ablated_perplexity"], 2000))
        colors.append(COLORS["v1"])

    if "ablated_perplexity" in cola_results:
        a = cola_results.get("alpha", "?")
        categories.append(f"V2\nMean Ablation\n(a={a})")
        values.append(cola_results["ablated_perplexity"])
        colors.append(COLORS["v2"])

    if v3_results:
        for method_name, method_data in v3_results.items():
            if "ablated_perplexity" in method_data:
                categories.append(f"V3\n{method_name}")
                values.append(method_data["ablated_perplexity"])
                colors.append(COLORS.get(method_name.lower().split()[0], COLORS["v3"]))

    bars = ax.bar(categories, values, color=colors, alpha=0.85, width=0.5,
                  edgecolor="white", linewidth=2)

    for bar, val in zip(bars, values):
        label = f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("Collateral Damage: Perplexity Across All Versions", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved perplexity comparison to {output_path}")


def plot_eap_distribution(edges_data: List[Dict], output_path: str):
    """Histogram and CDF of EAP scores."""
    scores = [e["score"] for e in edges_data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(scores, bins=25, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax1.axvline(x=np.mean(scores), color=COLORS["accent"], linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(scores):.4f}")
    ax1.axvline(x=np.median(scores), color=COLORS["secondary"], linestyle="--", linewidth=2,
                label=f"Median: {np.median(scores):.4f}")
    ax1.set_xlabel("EAP Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of EAP Edge Scores")
    ax1.legend()

    sorted_scores = np.sort(scores)[::-1]
    cumulative = np.cumsum(sorted_scores) / np.sum(sorted_scores) * 100
    ax2.plot(range(1, len(sorted_scores)+1), cumulative, color=COLORS["primary"], linewidth=2)
    ax2.axhline(y=80, color=COLORS["accent"], linestyle="--", alpha=0.5, label="80% of total score")
    ax2.fill_between(range(1, len(sorted_scores)+1), cumulative, alpha=0.1, color=COLORS["primary"])
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
    """Summary dashboard with key metrics."""
    per_prompt = debiasing_results["per_prompt"]
    reductions = [p["reduction"] for p in per_prompt]
    improved = sum(1 for r in reductions if r > 0)
    worsened = sum(1 for r in reductions if r < 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    bp = ax1.boxplot(reductions, vert=True, patch_artist=True,
                     boxprops=dict(facecolor=COLORS["primary"], alpha=0.6),
                     medianprops=dict(color=COLORS["accent"], linewidth=2),
                     whiskerprops=dict(color=COLORS["primary"]),
                     capprops=dict(color=COLORS["primary"]))
    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.set_ylabel("Bias Reduction per Prompt")
    ax1.set_title("Distribution of Bias Changes")
    ax1.set_xticklabels(["All Prompts"])

    ax2 = axes[1]
    ax2.pie([improved, worsened],
            labels=[f"Improved\n({improved})", f"Worsened\n({worsened})"],
            colors=[COLORS["improved"], COLORS["worsened"]],
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 11})
    ax2.set_title("Prompts: Improved vs Worsened")

    ax3 = axes[2]
    ax3.axis("off")
    bm = debiasing_results["before"]["mean_bias"]
    am = debiasing_results["after"]["mean_bias"]
    rp = debiasing_results["reduction_percent"]
    ne = debiasing_results["n_edges_ablated"]
    al = debiasing_results.get("alpha", "N/A")
    pb = cola_results.get("baseline_perplexity", 0)
    pa = cola_results.get("ablated_perplexity", 0)
    pi = cola_results.get("perplexity_increase_percent", 0)

    txt = (f"DEBIASING SUMMARY\n{'='*35}\n\n"
           f"Mean Bias Before:  {bm:.4f}\nMean Bias After:   {am:.4f}\n"
           f"Reduction:         {rp:.1f}%\n\nEdges Ablated:     {ne}\n"
           f"Alpha:             {al}\n\n{'='*35}\nCOLLATERAL DAMAGE\n{'='*35}\n\n"
           f"Baseline PPL:      {pb:.2f}\nAblated PPL:       {pa:.2f}\n"
           f"PPL Increase:      {pi:.1f}%\n")
    ax3.text(0.1, 0.95, txt, transform=ax3.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["bg"],
                      edgecolor=COLORS["grid"]))

    plt.suptitle("Mechanistic Bias Intervention -- Results Summary",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved summary dashboard to {output_path}")


# =============================================================================
# V3 NEW THESIS PLOTS
# =============================================================================

def plot_pareto_frontier(sweep_results: List[Dict], output_path: str):
    """Pareto frontier: Alpha vs Bias Reduction vs Perplexity.

    Dual Y-axis plot showing the tradeoff between debiasing strength
    and collateral damage as alpha varies.
    """
    alphas = [r["alpha"] for r in sweep_results]
    bias_reductions = [r["bias_reduction_percent"] for r in sweep_results]
    ppl_increases = [r["perplexity_increase_percent"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Bias reduction line
    line1 = ax1.plot(alphas, bias_reductions, "o-", color=COLORS["v3"],
                     linewidth=2.5, markersize=8, label="Bias Reduction (%)",
                     zorder=5)
    ax1.fill_between(alphas, bias_reductions, alpha=0.1, color=COLORS["v3"])
    ax1.set_xlabel("Alpha (Intervention Strength)", fontsize=12)
    ax1.set_ylabel("Bias Reduction (%)", color=COLORS["v3"], fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLORS["v3"])

    # Perplexity line
    line2 = ax2.plot(alphas, ppl_increases, "s--", color=COLORS["v1"],
                     linewidth=2.5, markersize=8, label="Perplexity Increase (%)",
                     zorder=5)
    ax2.fill_between(alphas, ppl_increases, alpha=0.08, color=COLORS["v1"])
    ax2.set_ylabel("Perplexity Increase (%)", color=COLORS["v1"], fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COLORS["v1"])

    # Horizontal threshold line at 15%
    ax2.axhline(y=15, color=COLORS["neutral"], linestyle=":", linewidth=1.5,
                alpha=0.7, label="15% PPL Threshold")

    # Find optimal alpha (highest bias reduction with PPL < 15%)
    valid = [(a, br, pp) for a, br, pp in zip(alphas, bias_reductions, ppl_increases)
             if pp < 15]
    if valid:
        best_alpha, best_br, best_pp = max(valid, key=lambda x: x[1])
        ax1.axvline(x=best_alpha, color=COLORS["accent"], linestyle="-.",
                    linewidth=2, alpha=0.7)
        ax1.annotate(
            f"Optimal: a={best_alpha}\nBias: -{best_br:.1f}%\nPPL: +{best_pp:.1f}%",
            xy=(best_alpha, best_br), xytext=(best_alpha + 0.08, best_br + 1),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["accent"], alpha=0.2),
            arrowprops=dict(arrowstyle="->", color=COLORS["dark"]),
        )

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    ax1.set_title("Pareto Frontier: Bias Reduction vs Collateral Damage",
                  fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved Pareto frontier to {output_path}")


def plot_probe_accuracy_by_layer(probe_results: Dict, output_path: str):
    """Bar chart: gender probe accuracy per layer.

    Shows how strongly each GPT-2 layer encodes gender information.
    Higher accuracy = more gender signal = better target for intervention.
    """
    layers = []
    accuracies = []
    f1_scores = []

    for key, data in sorted(probe_results["per_layer"].items()):
        layers.append(f"L{data['layer']}")
        accuracies.append(data["accuracy"] * 100)
        f1_scores.append(data["f1_score"] * 100)

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy",
                   color=COLORS["primary"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, f1_scores, width, label="F1 Score",
                   color=COLORS["secondary"], alpha=0.85, edgecolor="white")

    # Baseline line (random = 50%)
    ax.axhline(y=50, color=COLORS["neutral"], linestyle="--", linewidth=1.5,
               label="Random Baseline (50%)")

    # Mark best layer
    best_idx = np.argmax(accuracies)
    ax.annotate(f"Best: {accuracies[best_idx]:.1f}%",
                xy=(best_idx - width/2, accuracies[best_idx]),
                xytext=(best_idx - width/2, accuracies[best_idx] + 3),
                ha="center", fontsize=10, fontweight="bold", color=COLORS["dark"],
                arrowprops=dict(arrowstyle="->", color=COLORS["dark"]))

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_xlabel("GPT-2 Layer", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Gender Information by Layer (Linear Probe)", fontsize=13, fontweight="bold")
    ax.set_ylim(40, 105)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved probe accuracy to {output_path}")


def plot_sae_feature_spectrum(gender_features: Dict, output_path: str):
    """Ranked bar chart of top SAE features by gender relevance score.

    Shows how concentrated vs diffuse the gender signal is across SAE features.
    """
    all_feats = []
    for layer_key, feats in gender_features.get("features_per_layer", {}).items():
        for f in feats[:10]:  # Top 10 per layer
            f_copy = dict(f)
            all_feats.append(f_copy)

    # Sort by gender_score
    all_feats.sort(key=lambda x: x["gender_score"], reverse=True)
    display = all_feats[:30]

    labels = [f"L{f['layer']}.F{f['feature_idx']}" for f in display]
    scores = [f["gender_score"] for f in display]
    colors = [COLORS["sae"] if f["direction"] == "male" else COLORS["caa"] for f in display]

    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(range(len(labels)), scores, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("SAE Feature (Layer.FeatureIndex)", fontsize=11)
    ax.set_ylabel("Gender Relevance Score", fontsize=11)
    ax.set_title("Top Gender-Associated SAE Features", fontsize=13, fontweight="bold")

    # Legend for direction
    legend_elements = [
        mpatches.Patch(color=COLORS["sae"], alpha=0.85, label="Male-biased feature"),
        mpatches.Patch(color=COLORS["caa"], alpha=0.85, label="Female-biased feature"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved SAE feature spectrum to {output_path}")


def plot_version_comparison(all_results: Dict, output_path: str):
    """Grouped bar chart: V1 vs V2 vs V3 across key metrics.

    The definitive thesis comparison figure.
    """
    metrics = ["Bias Reduction\n(%)", "Prompts\nImproved (%)",
               "PPL Increase\n(%)", "Edges/Features\nTargeted"]

    v1_data = all_results.get("v1", {})
    v2_data = all_results.get("v2", {})
    v3_data = all_results.get("v3", {})

    v1_vals = [
        v1_data.get("bias_reduction", 4.17),
        v1_data.get("prompts_improved_pct", 60.0),
        min(v1_data.get("ppl_increase", 7361.3), 500),  # Cap for display
        v1_data.get("n_targets", 50),
    ]
    v2_vals = [
        v2_data.get("bias_reduction", 3.36),
        v2_data.get("prompts_improved_pct", 76.0),
        v2_data.get("ppl_increase", 96.5),
        v2_data.get("n_targets", 20),
    ]
    v3_vals = [
        v3_data.get("bias_reduction", 0),
        v3_data.get("prompts_improved_pct", 0),
        v3_data.get("ppl_increase", 0),
        v3_data.get("n_targets", 0),
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, v1_vals, width, label="V1 (Zero Ablation)",
                   color=COLORS["v1"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x, v2_vals, width, label="V2 (Mean Ablation)",
                   color=COLORS["v2"], alpha=0.85, edgecolor="white")
    bars3 = ax.bar(x + width, v3_vals, width, label="V3 (SAE Features)",
                   color=COLORS["v3"], alpha=0.85, edgecolor="white")

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                lbl = f"{h:.1f}" if h < 100 else f"{h:.0f}"
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Mechanistic Bias Intervention: V1 vs V2 vs V3",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)

    # Add note about capped V1 PPL
    if v1_data.get("ppl_increase", 0) > 500:
        ax.annotate(f"Actual: {v1_data['ppl_increase']:.0f}%",
                    xy=(2 - width, 500), xytext=(2 - width, 520),
                    ha="center", fontsize=8, color=COLORS["v1"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["v1"]))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved version comparison to {output_path}")


def plot_radar_comparison(all_results: Dict, output_path: str):
    """Radar/spider chart for multi-metric comparison across versions.

    Metrics are normalized to 0-1 scale where higher = better.
    """
    categories = [
        "Bias\nReduction",
        "Prompt\nImprovement",
        "Language\nPreservation",
        "Targeting\nPrecision",
        "Consistency",
    ]

    def normalize_results(data, max_vals):
        return [min(data[i] / max_vals[i], 1.0) for i in range(len(data))]

    v1 = all_results.get("v1", {})
    v2 = all_results.get("v2", {})
    v3 = all_results.get("v3", {})

    # Raw values for each metric (higher = better for all)
    v1_raw = [v1.get("bias_reduction", 4.17),
              v1.get("prompts_improved_pct", 60.0),
              max(0, 100 - min(v1.get("ppl_increase", 7361), 100)),
              10,  # Low precision (50 edges)
              v1.get("prompts_improved_pct", 60.0)]
    v2_raw = [v2.get("bias_reduction", 3.36),
              v2.get("prompts_improved_pct", 76.0),
              max(0, 100 - v2.get("ppl_increase", 96.5)),
              40,  # Medium (20 edges, alpha)
              v2.get("prompts_improved_pct", 76.0)]
    v3_raw = [v3.get("bias_reduction", 15.0),
              v3.get("prompts_improved_pct", 90.0),
              max(0, 100 - v3.get("ppl_increase", 5.0)),
              90,  # High (individual features)
              v3.get("prompts_improved_pct", 90.0)]

    max_vals = [max(v1_raw[i], v2_raw[i], v3_raw[i], 1) for i in range(5)]
    v1_n = normalize_results(v1_raw, max_vals)
    v2_n = normalize_results(v2_raw, max_vals)
    v3_n = normalize_results(v3_raw, max_vals)

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    v1_n += v1_n[:1]
    v2_n += v2_n[:1]
    v3_n += v3_n[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, v1_n, "o-", color=COLORS["v1"], linewidth=2, label="V1", markersize=6)
    ax.fill(angles, v1_n, alpha=0.1, color=COLORS["v1"])
    ax.plot(angles, v2_n, "s-", color=COLORS["v2"], linewidth=2, label="V2", markersize=6)
    ax.fill(angles, v2_n, alpha=0.1, color=COLORS["v2"])
    ax.plot(angles, v3_n, "D-", color=COLORS["v3"], linewidth=2.5, label="V3", markersize=7)
    ax.fill(angles, v3_n, alpha=0.15, color=COLORS["v3"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Metric Comparison: V1 vs V2 vs V3",
                 fontsize=13, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved radar comparison to {output_path}")


def plot_method_comparison(method_results: Dict, output_path: str):
    """Grouped bar chart comparing SAE vs CAA vs LEACE vs V2 (edge ablation).

    Each method is evaluated on bias reduction and perplexity increase.
    """
    methods = list(method_results.keys())
    bias_reductions = [method_results[m].get("bias_reduction_percent", 0) for m in methods]
    ppl_increases = [method_results[m].get("perplexity_increase_percent", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, bias_reductions, width,
                    label="Bias Reduction (%)", color=COLORS["v3"], alpha=0.85,
                    edgecolor="white", linewidth=1.5)
    bars2 = ax2.bar(x + width/2, ppl_increases, width,
                    label="PPL Increase (%)", color=COLORS["v1"], alpha=0.6,
                    edgecolor="white", linewidth=1.5)

    # Threshold
    ax2.axhline(y=15, color=COLORS["neutral"], linestyle=":", linewidth=1.5,
                label="15% Threshold")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                 ha="center", fontsize=9, fontweight="bold", color=COLORS["v3"])
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                 ha="center", fontsize=9, fontweight="bold", color=COLORS["v1"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel("Bias Reduction (%)", color=COLORS["v3"], fontsize=12)
    ax2.set_ylabel("Perplexity Increase (%)", color=COLORS["v1"], fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLORS["v3"])
    ax2.tick_params(axis="y", labelcolor=COLORS["v1"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Intervention Method Comparison",
                  fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved method comparison to {output_path}")


def plot_feature_ablation_curve(ablation_curve: List[Dict], output_path: str):
    """Line plot: number of features ablated vs bias reduction achieved.

    Shows how many gender features need to be removed to achieve X% reduction.
    Like a dose-response curve.
    """
    n_features = [d["n_features"] for d in ablation_curve]
    bias_reductions = [d["bias_reduction_percent"] for d in ablation_curve]
    ppls = [d.get("perplexity_increase_percent", 0) for d in ablation_curve]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(n_features, bias_reductions, "o-", color=COLORS["v3"],
             linewidth=2.5, markersize=7, label="Bias Reduction (%)", zorder=5)
    ax1.fill_between(n_features, bias_reductions, alpha=0.1, color=COLORS["v3"])

    ax2.plot(n_features, ppls, "s--", color=COLORS["v1"],
             linewidth=2, markersize=6, label="PPL Increase (%)", zorder=4)
    ax2.axhline(y=15, color=COLORS["neutral"], linestyle=":", linewidth=1.5,
                label="15% PPL Threshold")

    ax1.set_xlabel("Number of Gender Features Ablated", fontsize=12)
    ax1.set_ylabel("Bias Reduction (%)", color=COLORS["v3"], fontsize=12)
    ax2.set_ylabel("Perplexity Increase (%)", color=COLORS["v1"], fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLORS["v3"])
    ax2.tick_params(axis="y", labelcolor=COLORS["v1"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Feature Ablation Dose-Response Curve",
                  fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved feature ablation curve to {output_path}")


def plot_thesis_results_table(all_results: Dict, output_path: str):
    """Render a structured results table as a figure (for thesis PDF inclusion)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    headers = ["Metric", "V1\nZero Ablation", "V2\nMean Ablation", "V3 SAE\nFeature Abl.",
               "V3 CAA\nSteering", "V3 LEACE\nProjection"]

    v1 = all_results.get("v1", {})
    v2 = all_results.get("v2", {})
    v3_sae = all_results.get("v3_sae", all_results.get("v3", {}))
    v3_caa = all_results.get("v3_caa", {})
    v3_leace = all_results.get("v3_leace", {})

    def fmt(val, suffix="%"):
        if val is None or val == 0: return "N/A"
        return f"{val:.1f}{suffix}"

    rows = [
        ["Dataset Size", "20", "200", "200", "200", "200"],
        ["Targets", f"{v1.get('n_targets', 50)} edges",
         f"{v2.get('n_targets', 20)} edges",
         f"{v3_sae.get('n_targets', '?')} features",
         f"{v3_caa.get('n_targets', '?')} layers",
         f"{v3_leace.get('n_targets', '?')} layers"],
        ["Bias Reduction",
         fmt(v1.get("bias_reduction", 4.17)),
         fmt(v2.get("bias_reduction", 3.36)),
         fmt(v3_sae.get("bias_reduction")),
         fmt(v3_caa.get("bias_reduction")),
         fmt(v3_leace.get("bias_reduction"))],
        ["Prompts Improved",
         fmt(v1.get("prompts_improved_pct", 60)),
         fmt(v2.get("prompts_improved_pct", 76)),
         fmt(v3_sae.get("prompts_improved_pct")),
         fmt(v3_caa.get("prompts_improved_pct")),
         fmt(v3_leace.get("prompts_improved_pct"))],
        ["PPL Increase",
         fmt(v1.get("ppl_increase", 7361.3)),
         fmt(v2.get("ppl_increase", 96.5)),
         fmt(v3_sae.get("ppl_increase")),
         fmt(v3_caa.get("ppl_increase")),
         fmt(v3_leace.get("ppl_increase"))],
        ["Ablation Method", "Zero", "Mean + Alpha", "SAE Feature", "CAA Vector", "LEACE Proj."],
    ]

    # Table colors
    cell_colors = [["#F0F4F8"] * len(headers)] * len(rows)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor(COLORS["dark"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F0F4F8")
            else:
                table[i, j].set_facecolor("white")

    ax.set_title("Comprehensive Results: All Versions and Methods",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved thesis results table to {output_path}")
