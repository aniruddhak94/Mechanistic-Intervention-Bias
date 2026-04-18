#!/usr/bin/env python3
"""
08_generate_plots.py -- Generate all thesis-quality visualizations.

Reads all result JSON files and generates 12 publication-quality figures.
Supports V1, V2, and V3 results.

Usage (Kaggle):
    python scripts/08_generate_plots.py

Output:
    results/figures/*.png  (12+ figures)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import (
    # V2 plots
    plot_bias_comparison,
    plot_edge_heatmap,
    plot_perplexity_comparison,
    plot_eap_distribution,
    plot_bias_reduction_summary,
    # V3 plots
    plot_pareto_frontier,
    plot_probe_accuracy_by_layer,
    plot_sae_feature_spectrum,
    plot_version_comparison,
    plot_radar_comparison,
    plot_method_comparison,
    plot_feature_ablation_curve,
    plot_thesis_results_table,
)


RESULTS_DIR = os.path.join("results")
FIGURES_DIR = os.path.join("results", "figures")


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("  STEP 8: GENERATE ALL THESIS PLOTS")
    print("=" * 60)

    # ─── Load all result files ───
    debiasing = load_json(os.path.join(RESULTS_DIR, "debiasing_results.json"))
    edges = load_json(os.path.join(RESULTS_DIR, "top_edges_gender.json"))
    cola = load_json(os.path.join(RESULTS_DIR, "cola_evaluation.json"))
    v1_cola = load_json(os.path.join(RESULTS_DIR, "v1_cola_evaluation.json"))
    v3_results = load_json(os.path.join(RESULTS_DIR, "v3_debiasing_results.json"))
    pareto = load_json(os.path.join(RESULTS_DIR, "pareto_sweep.json"))
    probe = load_json(os.path.join(RESULTS_DIR, "v3_probe_results.json"))
    gender_feats = load_json(os.path.join(RESULTS_DIR, "gender_features.json"))

    plot_count = 0

    # ─── 1. Bias Comparison (V2) ───
    if debiasing:
        print("\n[1/12] Bias comparison bar chart...")
        plot_bias_comparison(debiasing, os.path.join(FIGURES_DIR, "bias_comparison.png"))
        plot_count += 1

    # ─── 2. Edge Heatmap ───
    if edges:
        print("[2/12] EAP edge heatmap...")
        plot_edge_heatmap(edges, 12, os.path.join(FIGURES_DIR, "edge_heatmap.png"))
        plot_count += 1

    # ─── 3. Perplexity Comparison (V1 vs V2 vs V3) ───
    if cola:
        print("[3/12] Perplexity comparison...")
        v3_ppl = {}
        if v3_results and "methods" in v3_results:
            for name, data in v3_results["methods"].items():
                if "ablated_perplexity" in data:
                    v3_ppl[name] = data
        plot_perplexity_comparison(
            cola, os.path.join(FIGURES_DIR, "perplexity_comparison.png"),
            v1_results=v1_cola, v3_results=v3_ppl if v3_ppl else None,
        )
        plot_count += 1

    # ─── 4. EAP Distribution ───
    if edges:
        print("[4/12] EAP score distribution...")
        plot_eap_distribution(edges, os.path.join(FIGURES_DIR, "eap_distribution.png"))
        plot_count += 1

    # ─── 5. Summary Dashboard ───
    if debiasing and cola:
        print("[5/12] Summary dashboard...")
        plot_bias_reduction_summary(
            debiasing, cola, os.path.join(FIGURES_DIR, "summary_dashboard.png")
        )
        plot_count += 1

    # ─── 6. Pareto Frontier ───
    if pareto and "sweep_results" in pareto:
        print("[6/12] Pareto frontier...")
        plot_pareto_frontier(
            pareto["sweep_results"],
            os.path.join(FIGURES_DIR, "pareto_frontier.png"),
        )
        plot_count += 1
    else:
        print("[6/12] SKIP pareto_frontier (no pareto_sweep.json)")

    # ─── 7. Probe Accuracy by Layer ───
    if probe or (v3_results and "probe_results" in v3_results):
        print("[7/12] Probe accuracy by layer...")
        probe_data = probe if probe else v3_results.get("probe_results", {})
        plot_probe_accuracy_by_layer(
            probe_data, os.path.join(FIGURES_DIR, "probe_accuracy.png"),
        )
        plot_count += 1
    else:
        print("[7/12] SKIP probe_accuracy (no probe results)")

    # ─── 8. SAE Feature Spectrum ───
    if gender_feats and "features_per_layer" in gender_feats:
        print("[8/12] SAE feature spectrum...")
        plot_sae_feature_spectrum(
            gender_feats, os.path.join(FIGURES_DIR, "sae_feature_spectrum.png"),
        )
        plot_count += 1
    else:
        print("[8/12] SKIP sae_feature_spectrum (no gender_features.json)")

    # ─── 9. Version Comparison (V1 vs V2 vs V3) ───
    print("[9/12] Version comparison...")
    all_version = {
        "v1": {
            "bias_reduction": 4.17,
            "prompts_improved_pct": 60.0,
            "ppl_increase": 7361.3,
            "n_targets": 50,
        },
        "v2": {
            "bias_reduction": 3.36,
            "prompts_improved_pct": 76.0,
            "ppl_increase": 96.5,
            "n_targets": 20,
        },
    }
    # Pull best V3 method results
    if v3_results and "methods" in v3_results:
        best_method = None
        best_score = -999
        for name, data in v3_results["methods"].items():
            score = data.get("bias_reduction_percent", 0) - data.get("perplexity_increase_percent", 0) * 0.1
            if score > best_score:
                best_score = score
                best_method = name
        if best_method:
            bm = v3_results["methods"][best_method]
            all_version["v3"] = {
                "bias_reduction": bm.get("bias_reduction_percent", 0),
                "prompts_improved_pct": bm.get("prompts_improved_percent", 0),
                "ppl_increase": bm.get("perplexity_increase_percent", 0),
                "n_targets": bm.get("n_targets", 0),
            }

    plot_version_comparison(all_version, os.path.join(FIGURES_DIR, "version_comparison.png"))
    plot_count += 1

    # ─── 10. Radar Chart ───
    print("[10/12] Radar comparison...")
    plot_radar_comparison(all_version, os.path.join(FIGURES_DIR, "radar_comparison.png"))
    plot_count += 1

    # ─── 11. Method Comparison ───
    if v3_results and "methods" in v3_results:
        print("[11/12] Method comparison...")
        method_data = {}
        # Add V2 as baseline comparison
        if cola and debiasing:
            method_data["V2 Edge\nAblation"] = {
                "bias_reduction_percent": debiasing.get("reduction_percent", 3.36),
                "perplexity_increase_percent": cola.get("perplexity_increase_percent", 96.5),
            }
        for name, data in v3_results["methods"].items():
            short = name.replace("Projection", "Proj.").replace("Ablation", "Abl.")
            method_data[short] = {
                "bias_reduction_percent": data.get("bias_reduction_percent", 0),
                "perplexity_increase_percent": data.get("perplexity_increase_percent", 0),
            }
        plot_method_comparison(
            method_data, os.path.join(FIGURES_DIR, "method_comparison.png"),
        )
        plot_count += 1
    else:
        print("[11/12] SKIP method_comparison (no V3 results)")

    # ─── 12. Thesis Results Table ───
    print("[12/12] Thesis results table...")
    table_data = dict(all_version)
    if v3_results and "methods" in v3_results:
        for name, data in v3_results["methods"].items():
            key = "v3_sae" if "SAE" in name else ("v3_caa" if "CAA" in name else "v3_leace")
            table_data[key] = {
                "bias_reduction": data.get("bias_reduction_percent", 0),
                "prompts_improved_pct": data.get("prompts_improved_percent", 0),
                "ppl_increase": data.get("perplexity_increase_percent", 0),
                "n_targets": data.get("n_targets", 0),
            }
    plot_thesis_results_table(table_data, os.path.join(FIGURES_DIR, "results_table.png"))
    plot_count += 1

    print(f"\n{'='*60}")
    print(f"  Generated {plot_count} plots in {FIGURES_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
