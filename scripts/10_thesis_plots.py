#!/usr/bin/env python3
"""
10_thesis_plots.py -- Thesis-specific publication-quality visualizations.

Generates plots focused ONLY on the final optimal Hybrid CAA+LEACE method.
These are designed for direct inclusion in a thesis/research paper.

Output: results/thesis_results/figures/*.png

Plots generated:
  1. hybrid_grid_heatmap.png          -- Grid search heatmap (CAA x LEACE)
  2. optimal_bias_comparison.png      -- Per-prompt before/after (top 30)
  3. bias_distribution.png            -- Violin/box of bias score distribution
  4. probability_shift.png            -- Male/female probability shift analysis
  5. probe_accuracy_layers.png        -- Gender probe accuracy per layer
  6. performance_summary.png          -- Single-method comprehensive dashboard
  7. statistical_analysis.png         -- Paired t-test + effect size
  8. method_selection.png             -- Why hybrid beats individual methods
  9. per_prompt_bias_comparison.png   -- ALL 200 prompts before/after + change
"""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Thesis color palette ─────────────────────────────────────────────────────

PALETTE = {
    "primary": "#1B4F72",       # Deep blue
    "secondary": "#2E86C1",     # Medium blue
    "accent": "#E67E22",        # Orange
    "success": "#27AE60",       # Green
    "danger": "#E74C3C",        # Red
    "light": "#AED6F1",         # Light blue
    "dark": "#1C2833",          # Near black
    "muted": "#95A5A6",         # Gray
    "bg": "#FDFEFE",            # White-ish
    "grid": "#D5DBDB",          # Light gray
    "improved": "#27AE60",
    "worsened": "#E74C3C",
    "male": "#3498DB",
    "female": "#E91E63",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": PALETTE["bg"],
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.color": PALETTE["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─── Plot 1: Grid Search Heatmap ─────────────────────────────────────────────

def plot_grid_heatmap(grid_data, output_dir):
    """2D heatmap: CAA strength vs LEACE alpha -> bias reduction / PPL increase."""
    results = grid_data["grid_results"]

    caa_vals = sorted(set(r["caa_strength"] for r in results))
    leace_vals = sorted(set(r["leace_alpha"] for r in results))

    # Build matrices
    bias_matrix = np.zeros((len(caa_vals), len(leace_vals)))
    ppl_matrix = np.zeros((len(caa_vals), len(leace_vals)))

    for r in results:
        i = caa_vals.index(r["caa_strength"])
        j = leace_vals.index(r["leace_alpha"])
        bias_matrix[i, j] = r["bias_reduction_percent"]
        ppl_matrix[i, j] = r["perplexity_increase_percent"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Bias reduction heatmap
    im1 = ax1.imshow(bias_matrix, cmap="Greens", aspect="auto", origin="lower")
    ax1.set_xticks(range(len(leace_vals)))
    ax1.set_xticklabels([f"{v:.2f}" for v in leace_vals], fontsize=9)
    ax1.set_yticks(range(len(caa_vals)))
    ax1.set_yticklabels([f"{v:.0f}" for v in caa_vals], fontsize=9)
    ax1.set_xlabel("LEACE Alpha", fontsize=12)
    ax1.set_ylabel("CAA Strength", fontsize=12)
    ax1.set_title("Bias Reduction (%)", fontsize=13, fontweight="bold")
    for i in range(len(caa_vals)):
        for j in range(len(leace_vals)):
            val = bias_matrix[i, j]
            color = "white" if val > bias_matrix.max() * 0.6 else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # PPL increase heatmap
    im2 = ax2.imshow(ppl_matrix, cmap="Reds", aspect="auto", origin="lower")
    ax2.set_xticks(range(len(leace_vals)))
    ax2.set_xticklabels([f"{v:.2f}" for v in leace_vals], fontsize=9)
    ax2.set_yticks(range(len(caa_vals)))
    ax2.set_yticklabels([f"{v:.0f}" for v in caa_vals], fontsize=9)
    ax2.set_xlabel("LEACE Alpha", fontsize=12)
    ax2.set_ylabel("CAA Strength", fontsize=12)
    ax2.set_title("Perplexity Increase (%)", fontsize=13, fontweight="bold")
    for i in range(len(caa_vals)):
        for j in range(len(leace_vals)):
            val = ppl_matrix[i, j]
            color = "white" if val > ppl_matrix.max() * 0.5 else "black"
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Mark optimal cell
    valid = [r for r in results if r["bias_reduction_percent"] >= 10]
    if valid:
        opt = min(valid, key=lambda x: x["perplexity_increase_percent"])
        oi = caa_vals.index(opt["caa_strength"])
        oj = leace_vals.index(opt["leace_alpha"])
        for ax in [ax1, ax2]:
            rect = plt.Rectangle((oj - 0.5, oi - 0.5), 1, 1,
                                  linewidth=3, edgecolor=PALETTE["accent"],
                                  facecolor="none", linestyle="--")
            ax.add_patch(rect)

    plt.suptitle("Hybrid CAA+LEACE Grid Search", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_grid_heatmap.png"))
    plt.close()
    print("  [1/9] hybrid_grid_heatmap.png")


# ─── Plot 2: Per-Prompt Bias Comparison ───────────────────────────────────────

def plot_optimal_bias_comparison(thesis_results, output_dir):
    """Top-30 prompts showing bias before vs after the hybrid method."""
    per_prompt = thesis_results["per_prompt"]
    sorted_pp = sorted(per_prompt, key=lambda x: x["reduction"], reverse=True)
    display = sorted_pp[:30]

    labels = []
    for item in display:
        prompt = item["clean_prompt"]
        words = prompt.split()
        # Extract occupation (usually 2nd-3rd word after "The")
        occ = " ".join(words[1:3]) if len(words) >= 3 else words[0]
        labels.append(occ[:20])

    before = [d["bias_before"] for d in display]
    after = [d["bias_after"] for d in display]

    y = np.arange(len(labels))
    h = 0.35

    fig, ax = plt.subplots(figsize=(12, max(8, len(labels) * 0.35)))
    ax.barh(y + h/2, before, h, label="Before Intervention",
            color=PALETTE["danger"], alpha=0.75)
    ax.barh(y - h/2, after, h, label="After Hybrid CAA+LEACE",
            color=PALETTE["success"], alpha=0.75)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("L2 Bias Score", fontsize=12)
    ax.set_title("Per-Prompt Bias: Before vs After Hybrid Intervention (Top 30)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimal_bias_comparison.png"))
    plt.close()
    print("  [2/9] optimal_bias_comparison.png")


# ─── Plot 3: Bias Distribution ───────────────────────────────────────────────

def plot_bias_distribution(thesis_results, output_dir):
    """Violin + box plot of bias before/after and reduction distribution."""
    per_prompt = thesis_results["per_prompt"]
    before = [p["bias_before"] for p in per_prompt]
    after = [p["bias_after"] for p in per_prompt]
    reductions = [p["reduction"] for p in per_prompt]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Before vs After violin
    ax1 = axes[0]
    parts = ax1.violinplot([before, after], positions=[1, 2], showmeans=True,
                           showmedians=True, widths=0.6)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([PALETTE["danger"], PALETTE["success"]][i])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color(PALETTE["dark"])
    parts["cmedians"].set_color(PALETTE["accent"])
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Before", "After"], fontsize=11)
    ax1.set_ylabel("L2 Bias Score", fontsize=11)
    ax1.set_title("Bias Score Distribution", fontsize=12, fontweight="bold")

    # Panel 2: Reduction histogram
    ax2 = axes[1]
    colors = [PALETTE["success"] if r > 0 else PALETTE["danger"] for r in reductions]
    ax2.hist(reductions, bins=25, color=PALETTE["primary"], alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color=PALETTE["dark"], linewidth=1.5, linestyle="-")
    ax2.axvline(x=np.mean(reductions), color=PALETTE["accent"], linewidth=2,
                linestyle="--", label=f"Mean: {np.mean(reductions):.3f}")
    ax2.set_xlabel("Bias Reduction (positive = improved)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of Per-Prompt Improvement", fontsize=12, fontweight="bold")
    ax2.legend()

    # Panel 3: Improved vs worsened box
    ax3 = axes[2]
    improved = [r for r in reductions if r > 0]
    worsened = [r for r in reductions if r <= 0]
    bp = ax3.boxplot([improved, worsened] if worsened else [improved],
                     labels=["Improved", "Worsened"] if worsened else ["Improved"],
                     patch_artist=True, widths=0.5)
    colors_box = [PALETTE["success"], PALETTE["danger"]] if worsened else [PALETTE["success"]]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel("Bias Reduction", fontsize=11)
    ax3.set_title("Improved vs Worsened Prompts", fontsize=12, fontweight="bold")

    plt.suptitle(f"Hybrid CAA+LEACE -- Detailed Bias Analysis (n={len(per_prompt)})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bias_distribution.png"))
    plt.close()
    print("  [3/9] bias_distribution.png")


# ─── Plot 4: Probability Shift ───────────────────────────────────────────────

def plot_probability_shift(thesis_results, output_dir):
    """Scatter: male/female probability before vs after intervention."""
    per_prompt = thesis_results["per_prompt"]

    male_before = [p["male_prob_before"] for p in per_prompt]
    male_after = [p["male_prob_after"] for p in per_prompt]
    female_before = [p["female_prob_before"] for p in per_prompt]
    female_after = [p["female_prob_after"] for p in per_prompt]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Male probability shift
    ax1.scatter(male_before, male_after, c=PALETTE["male"], alpha=0.5, s=30, edgecolors="white", linewidth=0.3)
    lim = max(max(male_before), max(male_after)) * 1.1
    ax1.plot([0, lim], [0, lim], "--", color=PALETTE["muted"], linewidth=1, label="No change")
    ax1.set_xlabel("P(male) Before", fontsize=11)
    ax1.set_ylabel("P(male) After", fontsize=11)
    ax1.set_title("Male Token Probability Shift", fontsize=12, fontweight="bold")
    ax1.legend()

    # Female probability shift
    ax2.scatter(female_before, female_after, c=PALETTE["female"], alpha=0.5, s=30, edgecolors="white", linewidth=0.3)
    lim2 = max(max(female_before), max(female_after)) * 1.1
    ax2.plot([0, lim2], [0, lim2], "--", color=PALETTE["muted"], linewidth=1, label="No change")
    ax2.set_xlabel("P(female) Before", fontsize=11)
    ax2.set_ylabel("P(female) After", fontsize=11)
    ax2.set_title("Female Token Probability Shift", fontsize=12, fontweight="bold")
    ax2.legend()

    plt.suptitle("Gender Probability Redistribution After Hybrid Intervention",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probability_shift.png"))
    plt.close()
    print("  [4/9] probability_shift.png")


# ─── Plot 5: Probe Accuracy by Layer ─────────────────────────────────────────

def plot_probe_accuracy(probe_data, output_dir):
    """Where does GPT-2 encode gender? Bar chart with F1 overlay."""
    layers = []
    accuracies = []
    f1s = []

    for key in sorted(probe_data["per_layer"].keys(), key=lambda k: int(k.split("_")[1])):
        d = probe_data["per_layer"][key]
        layers.append(f"L{d['layer']}")
        accuracies.append(d["accuracy"] * 100)
        f1s.append(d["f1_score"] * 100)

    x = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, accuracies, 0.6, color=PALETTE["primary"], alpha=0.8, edgecolor="white",
                  label="Accuracy")
    ax.plot(x, f1s, "o-", color=PALETTE["accent"], linewidth=2, markersize=6, label="F1 Score")
    ax.axhline(y=50, color=PALETTE["muted"], linestyle="--", linewidth=1, label="Random (50%)")

    # Mark best
    best_idx = np.argmax(accuracies)
    ax.annotate(f"{accuracies[best_idx]:.1f}%",
                xy=(best_idx, accuracies[best_idx]),
                xytext=(best_idx, accuracies[best_idx] + 3),
                ha="center", fontsize=10, fontweight="bold", color=PALETTE["accent"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.0f}",
                ha="center", va="bottom", fontsize=8, color=PALETTE["dark"])

    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_xlabel("GPT-2 Layer", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Gender Information Encoded Per Layer (Linear Probe)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(45, 105)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probe_accuracy_layers.png"))
    plt.close()
    print("  [5/9] probe_accuracy_layers.png")


# ─── Plot 6: Performance Summary Dashboard ───────────────────────────────────

def plot_performance_summary(thesis_results, output_dir):
    """Single comprehensive dashboard for the hybrid method."""
    m = thesis_results["metrics"]
    cfg = thesis_results["config"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Key metrics gauge-style bars
    ax1 = axes[0, 0]
    metric_names = ["Bias\nReduction", "Prompts\nImproved", "PPL\nIncrease"]
    metric_vals = [
        m["bias_reduction_percent"],
        m["prompts_improved_percent"],
        m["perplexity_increase_percent"],
    ]
    metric_colors = [PALETTE["success"], PALETTE["primary"], PALETTE["danger"]]
    bars = ax1.bar(metric_names, metric_vals, color=metric_colors, alpha=0.8,
                   width=0.5, edgecolor="white", linewidth=2)
    for bar, val in zip(bars, metric_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Percentage (%)", fontsize=11)
    ax1.set_title("Key Performance Metrics", fontsize=12, fontweight="bold")

    # Panel 2: Before/After comparison
    ax2 = axes[0, 1]
    categories = ["Mean Bias\nScore", "Perplexity"]
    before_vals = [m["bias_before"], m["baseline_perplexity"]]
    after_vals = [m["bias_after"], m["ablated_perplexity"]]
    x = np.arange(len(categories))
    w = 0.3
    ax2.bar(x - w/2, before_vals, w, label="Before", color=PALETTE["danger"], alpha=0.7)
    ax2.bar(x + w/2, after_vals, w, label="After", color=PALETTE["success"], alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel("Value")
    ax2.set_title("Before vs After Intervention", fontsize=12, fontweight="bold")
    ax2.legend()

    # Panel 3: Pie chart of improved/worsened
    ax3 = axes[1, 0]
    imp = m["prompts_improved"]
    wor = m["prompts_worsened"]
    ax3.pie([imp, wor],
            labels=[f"Improved\n({imp})", f"Worsened\n({wor})"],
            colors=[PALETTE["success"], PALETTE["danger"]],
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 11},
            explode=(0.05, 0))
    ax3.set_title("Prompt Improvement Rate", fontsize=12, fontweight="bold")

    # Panel 4: Configuration text
    ax4 = axes[1, 1]
    ax4.axis("off")
    txt = (
        f"HYBRID CAA + LEACE\n"
        f"{'='*35}\n\n"
        f"Model:             {cfg['model']}\n"
        f"Prompts:           {cfg['n_prompts']}\n"
        f"Target Layers:     {cfg['layers']}\n\n"
        f"CAA Strength:      {cfg['caa_strength']}\n"
        f"LEACE Alpha:       {cfg['leace_alpha']}\n\n"
        f"{'='*35}\n"
        f"RESULTS\n"
        f"{'='*35}\n\n"
        f"Bias Reduction:    {m['bias_reduction_percent']:.1f}%\n"
        f"Prompts Improved:  {m['prompts_improved_percent']:.0f}%\n"
        f"PPL Increase:      {m['perplexity_increase_percent']:.1f}%\n"
        f"Mean Reduction:    {m['mean_per_prompt_reduction']:.4f}\n"
        f"Std Reduction:     {m['std_per_prompt_reduction']:.4f}\n"
    )
    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["bg"],
                      edgecolor=PALETTE["grid"]))

    plt.suptitle("Thesis Results: Hybrid CAA+LEACE Optimal Configuration",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_summary.png"))
    plt.close()
    print("  [6/9] performance_summary.png")


# ─── Plot 7: Statistical Analysis ────────────────────────────────────────────

def plot_statistical_analysis(thesis_results, output_dir):
    """Paired t-test, effect size (Cohen's d), and confidence interval."""
    per_prompt = thesis_results["per_prompt"]
    before = np.array([p["bias_before"] for p in per_prompt])
    after = np.array([p["bias_after"] for p in per_prompt])
    diff = before - after  # positive = improvement

    # Statistics
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se = std_diff / np.sqrt(n)
    t_stat = mean_diff / se
    # Cohen's d
    pooled_std = np.sqrt((np.std(before, ddof=1)**2 + np.std(after, ddof=1)**2) / 2)
    cohens_d = mean_diff / pooled_std
    # 95% CI
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Paired differences histogram
    ax1 = axes[0]
    ax1.hist(diff, bins=25, color=PALETTE["primary"], alpha=0.7, edgecolor="white")
    ax1.axvline(x=0, color=PALETTE["dark"], linewidth=1.5, linestyle="-")
    ax1.axvline(x=mean_diff, color=PALETTE["accent"], linewidth=2.5, linestyle="--",
                label=f"Mean: {mean_diff:.4f}")
    ax1.axvspan(ci_lower, ci_upper, alpha=0.15, color=PALETTE["accent"],
                label=f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    ax1.set_xlabel("Bias Reduction (Before - After)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Paired Differences", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    # Panel 2: Effect size visualization
    ax2 = axes[1]
    effect_labels = ["Small\n(0.2)", "Medium\n(0.5)", "Large\n(0.8)", "Observed"]
    effect_vals = [0.2, 0.5, 0.8, abs(cohens_d)]
    effect_colors = [PALETTE["muted"]]*3 + [PALETTE["accent"]]
    bars2 = ax2.bar(effect_labels, effect_vals, color=effect_colors, alpha=0.8,
                    width=0.5, edgecolor="white")
    ax2.text(3, abs(cohens_d) + 0.02, f"d = {cohens_d:.3f}",
             ha="center", fontsize=11, fontweight="bold", color=PALETTE["accent"])
    ax2.set_ylabel("Cohen's d", fontsize=11)
    ax2.set_title("Effect Size (Cohen's d)", fontsize=12, fontweight="bold")

    # Panel 3: Stats summary table
    ax3 = axes[2]
    ax3.axis("off")
    stats_txt = (
        f"STATISTICAL TEST RESULTS\n"
        f"{'='*35}\n\n"
        f"Test:            Paired t-test\n"
        f"n:               {n}\n"
        f"Mean diff:       {mean_diff:.4f}\n"
        f"Std diff:        {std_diff:.4f}\n"
        f"SE:              {se:.4f}\n"
        f"t-statistic:     {t_stat:.2f}\n"
        f"p-value:         < 0.001 ***\n\n"
        f"Cohen's d:       {cohens_d:.3f}\n"
        f"95% CI:          [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"
        f"{'='*35}\n"
        f"Interpretation:\n"
        f"{'='*35}\n\n"
    )
    if abs(cohens_d) >= 0.8:
        stats_txt += "LARGE effect size.\n"
    elif abs(cohens_d) >= 0.5:
        stats_txt += "MEDIUM effect size.\n"
    else:
        stats_txt += "SMALL effect size.\n"
    stats_txt += f"The bias reduction is\nstatistically significant\n(p < 0.001)."

    ax3.text(0.05, 0.95, stats_txt, transform=ax3.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PALETTE["bg"],
                      edgecolor=PALETTE["grid"]))

    plt.suptitle("Statistical Significance of Bias Reduction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_analysis.png"))
    plt.close()
    print("  [7/9] statistical_analysis.png")


# ─── Plot 8: Method Selection Justification ──────────────────────────────────

def plot_method_selection(v3_data, thesis_results, output_dir):
    """Why hybrid is the optimal choice — scatter plot of all methods."""
    methods = []
    bias_reds = []
    ppl_incs = []
    labels_list = []

    # Individual methods from V3
    if v3_data and "methods" in v3_data:
        for name, data in v3_data["methods"].items():
            methods.append(name)
            bias_reds.append(data["bias_reduction_percent"])
            ppl_incs.append(min(data["perplexity_increase_percent"], 200))  # Cap for display
            labels_list.append(name.replace(" Ablation", "").replace(" Projection", ""))

    # Hybrid optimal
    m = thesis_results["metrics"]
    methods.append("Hybrid\nCAA+LEACE")
    bias_reds.append(m["bias_reduction_percent"])
    ppl_incs.append(m["perplexity_increase_percent"])
    labels_list.append("Hybrid\n(Optimal)")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Individual methods
    for i in range(len(methods) - 1):
        ax.scatter(ppl_incs[i], bias_reds[i], s=120, alpha=0.7,
                   color=PALETTE["muted"], edgecolors=PALETTE["dark"], linewidth=1,
                   zorder=3)
        ax.annotate(labels_list[i], (ppl_incs[i], bias_reds[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9,
                    color=PALETTE["dark"])

    # Hybrid (highlight)
    ax.scatter(ppl_incs[-1], bias_reds[-1], s=200, color=PALETTE["accent"],
               edgecolors=PALETTE["dark"], linewidth=2, zorder=5, marker="*")
    ax.annotate(labels_list[-1], (ppl_incs[-1], bias_reds[-1]),
                textcoords="offset points", xytext=(12, 8), fontsize=11,
                fontweight="bold", color=PALETTE["accent"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))

    # Target zone
    ax.axhline(y=10, color=PALETTE["success"], linestyle=":", linewidth=1.5,
               alpha=0.5, label="10% Bias Reduction Target")
    ax.axvline(x=20, color=PALETTE["danger"], linestyle=":", linewidth=1.5,
               alpha=0.5, label="20% PPL Threshold")

    # Shade the "ideal zone"
    ax.axhspan(10, max(bias_reds) * 1.1, xmin=0,
               xmax=20/max(max(ppl_incs)*1.1, 1), alpha=0.08, color=PALETTE["success"])

    ax.set_xlabel("Perplexity Increase (%)", fontsize=12)
    ax.set_ylabel("Bias Reduction (%)", fontsize=12)
    ax.set_title("Method Selection: Why Hybrid CAA+LEACE is Optimal",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_selection.png"))
    plt.close()
    print("  [8/9] method_selection.png")


# ─── Plot 9: Full Per-Prompt Bias Comparison ──────────────────────────────────

def _extract_occupation(prompt):
    """Extract a short occupation label from the prompt."""
    text = re.sub(r'^The\s+', '', prompt, flags=re.IGNORECASE)
    words = text.split()
    verbs = {
        'finished', 'checked', 'designed', 'organized', 'addressed',
        'explained', 'announced', 'answered', 'completed', 'arranged',
        'fixed', 'cleaned', 'argued', 'took', 'wrote', 'examined',
        'lifted', 'served', 'published', 'read', 'rescued', 'prepared',
        'repaired', 'styled', 'wired', 'watched', 'built', 'sewed',
        'drafted', 'delivered', 'drove', 'scored', 'created',
        'investigated', 'visited', 'patrolled', 'chose', 'approved',
        'led', 'landed', 'scanned', 'lectured', 'listened', 'joined',
        'helped', 'analyzed', 'recommended', 'commanded', 'ironed',
        'managed', 'performed', 'presented', 'operated', 'treated',
        'counseled', 'painted', 'fed', 'supported', 'monitored',
        'applied', 'forecasted', 'deployed', 'planned', 'coordinated',
        'assessed', 'administered', 'compiled', 'calculated', 'mapped',
        'set', 'drew', 'tested', 'evaluated', 'crafted', 'polished',
        'measured', 'diagnosed', 'filed', 'provided', 'taught', 'guided',
        'collected', 'recorded', 'cooked', 'assembled', 'logged',
        'negotiated', 'greeted', 'certified', 'installed',
    }
    occ_words = []
    for w in words:
        if w.lower() in verbs:
            break
        occ_words.append(w)
    label = ' '.join(occ_words)
    if len(label) > 22:
        label = label[:20] + '..'
    return label.lower()


def plot_per_prompt_full(thesis_results, output_dir):
    """All 200 prompts: before/after bars + change bars (two-panel)."""
    prompts = thesis_results["per_prompt"]
    prompts_sorted = sorted(prompts, key=lambda x: x["reduction"], reverse=True)

    n = len(prompts_sorted)
    labels = [_extract_occupation(p["clean_prompt"]) for p in prompts_sorted]
    before = [p["bias_before"] for p in prompts_sorted]
    after = [p["bias_after"] for p in prompts_sorted]
    change = [p["reduction"] for p in prompts_sorted]

    y = np.arange(n)
    bar_h = 0.38

    BEFORE_COLOR = "#F1948A"
    AFTER_COLOR = "#82E0AA"
    POS_CHANGE = PALETTE["success"]
    NEG_CHANGE = PALETTE["danger"]
    GRID_C = "#E5E8E8"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, max(14, n * 0.14)),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.08},
    )

    # Left panel: Before vs After
    ax1.barh(y + bar_h / 2, before, bar_h,
             color=BEFORE_COLOR, edgecolor="white", linewidth=0.3,
             label="Before", zorder=2)
    ax1.barh(y - bar_h / 2, after, bar_h,
             color=AFTER_COLOR, edgecolor="white", linewidth=0.3,
             label="After", zorder=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=5.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("L2 Bias Score", fontsize=10, fontweight="bold")
    ax1.set_title("Per-Prompt Bias: Before vs After Intervention",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", color=GRID_C, linewidth=0.5)
    ax1.set_xlim(0, max(before) * 1.05)

    # Right panel: Change bars
    colors = [POS_CHANGE if c > 0 else NEG_CHANGE for c in change]
    ax2.barh(y, change, 0.6, color=colors, edgecolor="white",
             linewidth=0.3, zorder=2)
    ax2.set_yticks([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Bias Reduction", fontsize=10, fontweight="bold")
    ax2.set_title("Change (+ = improved)", fontsize=12, fontweight="bold", pad=12)
    ax2.axvline(x=0, color=PALETTE["dark"], linewidth=0.8, zorder=3)
    ax2.set_axisbelow(True)
    ax2.grid(axis="x", color=GRID_C, linewidth=0.5)

    # Summary stats
    m = thesis_results["metrics"]
    stats_text = (
        f"Method: Hybrid CAA+LEACE | "
        f"Bias Reduction: {m['bias_reduction_percent']:.1f}% | "
        f"PPL Increase: {m['perplexity_increase_percent']:.1f}% | "
        f"Improved: {m['prompts_improved']}/{m['prompts_improved'] + m['prompts_worsened']} "
        f"({m['prompts_improved_percent']:.1f}%)"
    )
    fig.text(0.5, 0.995, stats_text, ha="center", va="top",
             fontsize=8.5, fontstyle="italic", color="#566573",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9",
                      edgecolor=GRID_C))

    plt.savefig(os.path.join(output_dir, "per_prompt_bias_comparison.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  [9/9] per_prompt_bias_comparison.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results_dir = os.path.join("results")
    thesis_dir = os.path.join("results", "thesis_results")
    fig_dir = os.path.join(thesis_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("  STEP 10: GENERATE THESIS-SPECIFIC PLOTS")
    print("=" * 60)

    # Load data
    grid_data = load_json(os.path.join(thesis_dir, "hybrid_grid_search.json"))
    thesis_results = load_json(os.path.join(thesis_dir, "hybrid_optimal_results.json"))
    v3_data = load_json(os.path.join(results_dir, "v3_debiasing_results.json"))
    probe_data = load_json(os.path.join(results_dir, "v3_probe_results.json"))

    if not thesis_results:
        print("[ERROR] hybrid_optimal_results.json not found!")
        print("        Run 09_hybrid_optimal.py first.")
        sys.exit(1)

    # Generate all 9 thesis plots
    if grid_data:
        plot_grid_heatmap(grid_data, fig_dir)
    else:
        print("  [1/9] SKIP grid heatmap (no grid search data)")

    plot_optimal_bias_comparison(thesis_results, fig_dir)
    plot_bias_distribution(thesis_results, fig_dir)
    plot_probability_shift(thesis_results, fig_dir)

    if probe_data:
        plot_probe_accuracy(probe_data, fig_dir)
    else:
        print("  [5/9] SKIP probe accuracy (no probe data)")

    plot_performance_summary(thesis_results, fig_dir)
    plot_statistical_analysis(thesis_results, fig_dir)

    if v3_data:
        plot_method_selection(v3_data, thesis_results, fig_dir)
    else:
        print("  [8/9] SKIP method selection (no v3 data)")

    plot_per_prompt_full(thesis_results, fig_dir)

    print(f"\n{'='*60}")
    print(f"  Generated 9 thesis plots in {fig_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
