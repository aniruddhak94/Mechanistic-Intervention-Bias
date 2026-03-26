---
description: How to run the Mechanistic Intervention Bias experiments on Kaggle or Google Colab
---

# Mechanistic Intervention for Bias Reduction — Experiment Workflow

## Paper Summary

This project implements the methodology from the **Mechanistic Interpretability of Bias** paper. The core idea is:

1. **Language models encode societal biases** in specific internal circuits (connections between attention heads and MLPs across layers).
2. **Edge Attribution Patching (EAP)** is a technique to precisely locate which internal "edges" (connections) in the model cause biased predictions.
3. **Targeted intervention** on just those edges can reduce bias while preserving the model's general language abilities.

### Key Terminology
- **Clean prompt**: A sentence containing a biased word (e.g., "The doctor finished surgery and everyone praised")
- **Corrupted prompt**: The same sentence with the biased word replaced by a neutral alternative (e.g., "The broadcaster finished surgery and everyone praised")
- **Edge**: A connection between a source component (attention head or MLP in layer i) and a destination component (attention head or MLP in layer j, where j > i)
- **EAP Score**: For each edge, the product of (gradient of bias metric w.r.t. destination activation) × (difference in source activation between clean and corrupted runs). High score = the edge contributes heavily to bias.
- **Ablation/Intervention**: Replacing the activation flowing through a specific edge with the activation from the corrupted (neutral) run during inference.

### The L2 Bias Metric
The paper uses an L2-norm metric to quantify bias:
1. Get the model's logit output for the last token position
2. Extract logits for male-associated tokens (he, him, his, man, etc.) and female-associated tokens (she, her, hers, woman, etc.)
3. Compute: `bias_score = ||log_softmax(male_logits) - log_softmax(female_logits)||_2`
4. A higher score means the model differentiates more strongly between male and female continuations — i.e., more bias.

---

## Prerequisites

- A **Google Colab** (free T4 GPU) or **Kaggle** (free P100 GPU) account
- Python 3.8+
- Internet access (to download model weights on first run)

---

## Step-by-Step Workflow

### Step 0: Setup Environment
// turbo-all

```bash
# Clone the repository (or upload files manually)
git clone https://github.com/<your-username>/Mechanistic-Intervention-Bias.git
cd Mechanistic-Intervention-Bias

# Install dependencies
pip install -r requirements.txt
```

> **Colab note:** You may already have `torch` and `transformers` installed. The key extra package is `transformer_lens`.

---

### Step 1: Run Baseline Bias Measurement

This step loads GPT-2 and measures how biased it is *before* any intervention.

```bash
python scripts/01_run_baseline.py --model gpt2 --dataset data/gender_bias.json
```

**What to expect:**
- Prints per-prompt bias scores
- Saves aggregate results to `results/baseline_scores.json`
- Typical bias scores for GPT-2: 0.5–2.0 range (higher = more biased)

---

### Step 2: Find Bias Circuits via EAP

This step runs Edge Attribution Patching across all prompt pairs to find the most bias-causing edges.

```bash
# For gender bias
python scripts/02_find_circuits.py --model gpt2 --dataset data/gender_bias.json --top_k 50 --output results/top_edges_gender.json

# For demographic bias
python scripts/02_find_circuits.py --model gpt2 --dataset data/demographic_bias.json --top_k 50 --output results/top_edges_demographic.json
```

**What to expect:**
- Takes 2-5 minutes on a free GPU
- Saves top-50 edges as JSON with their EAP scores
- Each edge is described as `(source_layer, source_head_or_mlp, dest_layer, dest_head_or_mlp)`

---

### Step 3: Debias via Intervention

Using the top edges from Step 2, this step patches them during inference and measures the new (lower) bias score.

```bash
python scripts/03_run_debiasing.py --model gpt2 --dataset data/gender_bias.json --edges results/top_edges_gender.json
```

**What to expect:**
- Prints before/after bias scores for each prompt
- Shows aggregate bias reduction percentage
- Saves results to `results/debiasing_results.json`
- Target: ≥30% bias reduction

---

### Step 4: Check for Collateral Damage (CoLA Evaluation)

This step ensures the model still understands English grammar after intervention.

```bash
python scripts/04_evaluate_cola.py --model gpt2 --edges results/top_edges_gender.json
```

**What to expect:**
- Downloads the CoLA dataset automatically
- Evaluates model accuracy on grammatical acceptability
- Compares accuracy with and without intervention
- Acceptable degradation: < 5% accuracy drop

---

### Step 5: Generate Visualizations (Notebooks)

Open the Jupyter notebooks for publication-quality visualizations:

1. **`notebooks/1_data_exploration.ipynb`** — Dataset statistics and example pairs
2. **`notebooks/2_layer_heatmaps.ipynb`** — EAP score heatmaps across layers/heads (purple heatmaps like Fig 2 in the paper)
3. **`notebooks/3_overlap_analysis.ipynb`** — Edge overlap matrices between gender and demographic bias circuits

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA Out of Memory | Use `gpt2` (small) instead of `gpt2-large`; reduce `--top_k` |
| Slow EAP computation | Ensure GPU is enabled in Colab (Runtime → Change runtime type → GPU) |
| TransformerLens import error | Run `pip install transformer_lens --upgrade` |
| CoLA dataset download fails | Check internet connection; Colab sometimes blocks HuggingFace |
