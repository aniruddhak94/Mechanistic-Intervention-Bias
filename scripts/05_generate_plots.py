#!/usr/bin/env python3
"""
05_generate_plots.py -- Generate all visualization plots from results.

Usage:
    python scripts/05_generate_plots.py

Reads JSON results from results/ and generates PNG figures in results/figures/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import (
    plot_bias_comparison,
    plot_edge_heatmap,
    plot_perplexity_comparison,
    plot_eap_distribution,
    plot_bias_reduction_summary,
)


def load_json(path):
    """Load a JSON file, return None if not found."""
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("  STEP 5: GENERATING VISUALIZATION PLOTS")
    print("=" * 60)

    figures_dir = "results/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # Load result files
    debiasing = load_json("results/debiasing_results.json")
    edges = load_json("results/top_edges_gender.json")
    cola = load_json("results/cola_evaluation.json")

    # Try loading V1 cola results for comparison
    v1_cola = load_json("results/v1_cola_evaluation.json")

    # 1. Bias comparison bar chart
    if debiasing:
        print("\n[1/5] Generating bias comparison chart...")
        plot_bias_comparison(debiasing, f"{figures_dir}/bias_comparison.png")
    else:
        print("\n[1/5] Skipping bias comparison (no debiasing_results.json)")

    # 2. Edge circuit heatmap
    if edges:
        print("\n[2/5] Generating edge heatmap...")
        plot_edge_heatmap(edges, n_layers=12, output_path=f"{figures_dir}/edge_heatmap.png")
    else:
        print("\n[2/5] Skipping edge heatmap (no top_edges_gender.json)")

    # 3. Perplexity comparison
    if cola:
        print("\n[3/5] Generating perplexity comparison...")
        plot_perplexity_comparison(cola, f"{figures_dir}/perplexity_comparison.png",
                                   v1_results=v1_cola)
    else:
        print("\n[3/5] Skipping perplexity comparison (no cola_evaluation.json)")

    # 4. EAP score distribution
    if edges:
        print("\n[4/5] Generating EAP score distribution...")
        plot_eap_distribution(edges, f"{figures_dir}/eap_distribution.png")
    else:
        print("\n[4/5] Skipping EAP distribution (no top_edges_gender.json)")

    # 5. Summary dashboard
    if debiasing and cola:
        print("\n[5/5] Generating summary dashboard...")
        plot_bias_reduction_summary(debiasing, cola, f"{figures_dir}/summary_dashboard.png")
    else:
        print("\n[5/5] Skipping summary dashboard (missing results)")

    print(f"\n[OK] All plots saved to {figures_dir}/")


if __name__ == "__main__":
    main()
