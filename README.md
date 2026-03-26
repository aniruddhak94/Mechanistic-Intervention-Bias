# Mechanistic Intervention for Bias Reduction in Language Models

This project uses **Edge Attribution Patching (EAP)** to identify and deactivate bias-causing circuits in GPT-2, following the methodology from the *Mechanistic Interpretability Bias* research paper.

## Overview

| Step | What it does | Script |
|------|-------------|--------|
| 1 | Measure baseline bias | `scripts/01_run_baseline.py` |
| 2 | Find bias circuits via EAP | `scripts/02_find_circuits.py` |
| 3 | Intervene & debias | `scripts/03_run_debiasing.py` |
| 4 | Evaluate collateral damage (CoLA) | `scripts/04_evaluate_cola.py` |

## Quick Start (Google Colab / Kaggle)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Mechanistic-Intervention-Bias.git
cd Mechanistic-Intervention-Bias

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python scripts/01_run_baseline.py --model gpt2 --dataset data/gender_bias.json
python scripts/02_find_circuits.py --model gpt2 --dataset data/gender_bias.json --top_k 50
python scripts/03_run_debiasing.py --model gpt2 --dataset data/gender_bias.json --edges results/top_edges_gender.json
python scripts/04_evaluate_cola.py --model gpt2 --edges results/top_edges_gender.json
```

## Project Structure

```
Mechanistic-Intervention-Bias/
├── data/                        # Bias datasets (clean/corrupted prompt pairs)
│   ├── gender_bias.json
│   ├── demographic_bias.json
│   └── downstream_tasks/        # CoLA and other evaluation sets
├── src/                         # Core library
│   ├── data_utils.py            # Dataset loading & prompt pair creation
│   ├── model_loader.py          # GPT-2 loading via TransformerLens
│   ├── baseline_scoring.py      # L2 bias metric computation
│   ├── eap_algorithm.py         # Edge Attribution Patching implementation
│   └── intervention.py          # Activation patching for debiasing
├── scripts/                     # Experiment runner scripts
├── notebooks/                   # Visualization notebooks
├── results/                     # Output: top edges, figures, scores
├── requirements.txt
└── README.md
```

## Key Concepts

### Edge Attribution Patching (EAP)
EAP identifies which *edges* (connections between attention heads / MLPs across layers) are responsible for biased predictions. It works by:
1. Running the model on a **clean** (biased) prompt and a **corrupted** (neutral) prompt
2. Computing gradients of a bias metric w.r.t. intermediate activations
3. Multiplying gradients by the activation difference (clean − corrupted) to score each edge
4. Ranking edges by score — the top edges are the "bias circuit"

### Intervention (Debiasing)
Once top edges are identified, we **patch** them during inference: we replace their activations with those from the corrupted (neutral) run. This "turns off" the bias circuit without retraining.

### Collateral Damage Check
We evaluate the model on the **CoLA** (Corpus of Linguistic Acceptability) task to ensure that general language ability is preserved after intervention.

## Hardware Requirements

| Platform | GPU | RAM | Status |
|----------|-----|-----|--------|
| Google Colab (free) | T4 16GB | 12GB | ✅ Works |
| Kaggle (free) | P100 16GB | 13GB | ✅ Works |
| CPU only | — | 8GB+ | ⚠️ Slow but works |

**Default model:** `gpt2` (124M params, ~500MB). Change to `gpt2-medium` or `gpt2-large` if you have more GPU memory.

## License

MIT
