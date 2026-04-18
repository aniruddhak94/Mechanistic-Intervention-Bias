# Mechanistic Intervention for Bias Reduction in Language Models

This project uses **Edge Attribution Patching (EAP)** and **Sparse Autoencoders (SAEs)** to identify and deactivate bias-causing circuits in GPT-2, following state-of-the-art Mechanistic Interpretability methodology.

## Overview

### Pipeline Steps

| Step | What it does | Script | Version |
|------|-------------|--------|---------|
| 1 | Measure baseline bias | `scripts/01_run_baseline.py` | V1+ |
| 2 | Find bias circuits via EAP | `scripts/02_find_circuits.py` | V1+ |
| 3 | Intervene & debias (edge ablation) | `scripts/03_run_debiasing.py` | V2+ |
| 4 | Evaluate collateral damage (CoLA) | `scripts/04_evaluate_cola.py` | V2+ |
| 5 | **SAE gender feature discovery** | `scripts/05_sae_discovery.py` | **V3** |
| 6 | **Pareto alpha sweep** | `scripts/06_pareto_sweep.py` | **V3** |
| 7 | **V3 debiasing (SAE + CAA + LEACE)** | `scripts/07_run_v3_debiasing.py` | **V3** |
| 8 | **Generate all thesis plots** | `scripts/08_generate_plots.py` | **V3** |

### Version Evolution

| Version | Method | Bias Reduction | PPL Increase | Key Innovation |
|---------|--------|----------------|--------------|----------------|
| **V1** | Zero ablation, 50 edges | 4.17% | 7,361% | Proof-of-concept |
| **V2** | Mean ablation + alpha blending | 3.36% | 96.5% | Surgical intervention |
| **V3** | SAE features, CAA, LEACE | TBD | TBD | Feature-level debiasing |

## Quick Start (Kaggle)

```bash
# 1. Clone the repo
git clone https://github.com/aniruddhak94/Mechanistic-Intervention-Bias.git
cd Mechanistic-Intervention-Bias

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run V2 pipeline (edge-level)
python scripts/01_run_baseline.py --model gpt2 --dataset data/gender_bias.json
python scripts/02_find_circuits.py --model gpt2 --dataset data/gender_bias.json --top_k 20 --min_layer 1
python scripts/03_run_debiasing.py --model gpt2 --dataset data/gender_bias.json --edges results/top_edges_gender.json --alpha 0.3
python scripts/04_evaluate_cola.py --model gpt2 --edges results/top_edges_gender.json --alpha 0.3

# 4. Run V3 pipeline (feature-level)
python scripts/05_sae_discovery.py --model gpt2 --dataset data/gender_bias.json
python scripts/06_pareto_sweep.py --model gpt2 --dataset data/gender_bias.json
python scripts/07_run_v3_debiasing.py --model gpt2 --dataset data/gender_bias.json
python scripts/08_generate_plots.py
```

## Project Structure

```
Mechanistic-Intervention-Bias/
├── data/
│   └── gender_bias.json         # 200 gendered prompt pairs
├── src/                         # Core library
│   ├── data_utils.py            # Dataset loading & prompt pair creation
│   ├── model_loader.py          # GPT-2 loading via TransformerLens
│   ├── baseline_scoring.py      # L2 + KL-divergence bias metrics
│   ├── eap_algorithm.py         # Edge Attribution Patching (EAP)
│   ├── intervention.py          # V2: Edge ablation with alpha blending
│   ├── sae_analysis.py          # V3: SAE feature discovery & ablation
│   ├── linear_probe.py          # V3: Linear probing, CAA, LEACE
│   └── visualization.py         # 12 thesis-quality plot types
├── scripts/                     # Pipeline runner scripts (01-08)
├── results/                     # Output: JSON results + figures
│   ├── figures/                 # All generated plots
│   ├── v1_run_report.md         # V1 results documentation
│   ├── v2_run_report.md         # V2 results documentation
│   └── summary_v1_vs_v2.md      # Detailed V1 vs V2 comparison
├── notebooks/                   # Exploration notebooks
├── requirements.txt
└── README.md
```

## Key Concepts

### Edge Attribution Patching (EAP)
EAP identifies which internal connections (edges) between layers are responsible for biased predictions by computing gradients of a bias metric w.r.t. activation differences.

### Sparse Autoencoders (SAEs) — V3
SAEs decompose polysemantic MLP activations into monosemantic features. Instead of ablating an entire edge (which removes bias AND grammar), V3 ablates only the specific SAE features associated with gender.

### V3 Intervention Methods

| Method | Technique | Reference |
|--------|-----------|-----------|
| **SAE Feature Ablation** | Zero gender features in SAE space | Cunningham et al., 2023 |
| **CAA Steering** | Subtract gender direction from residual stream | Rimsky et al., 2024 |
| **LEACE Projection** | Project out gender subspace | Belrose et al., 2023 |

## Hardware Requirements

| Platform | GPU | RAM | Status |
|----------|-----|-----|--------|
| Kaggle (free) | T4 16GB | 13GB | ✅ Recommended |
| Google Colab (free) | T4 16GB | 12GB | ✅ Works |
| CPU only | — | 8GB+ | ⚠️ Slow |

**Model:** GPT-2 Small (124M params, 12 layers, 768 d_model)

## Thesis Visualizations (12 Plots)

| # | Plot | Description |
|---|------|-------------|
| 1 | Bias Comparison | Per-prompt before/after bar chart |
| 2 | Edge Heatmap | Layer-to-layer EAP score grid |
| 3 | Perplexity Comparison | V1 vs V2 vs V3 perplexity bars |
| 4 | EAP Distribution | Histogram + CDF of EAP scores |
| 5 | Summary Dashboard | Box plot + pie chart + metrics |
| 6 | **Pareto Frontier** | Alpha vs bias vs perplexity tradeoff |
| 7 | **Probe Accuracy** | Gender info by layer (linear probe) |
| 8 | **SAE Feature Spectrum** | Top gender features ranked |
| 9 | **Version Comparison** | V1/V2/V3 grouped bars |
| 10 | **Radar Chart** | Multi-metric spider comparison |
| 11 | **Method Comparison** | SAE vs CAA vs LEACE |
| 12 | **Results Table** | Thesis-ready table figure |

## License

MIT
