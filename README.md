# Enhancing Clinical Decision Support: Benchmarking ML & Transformer Models for Symptom-Based Diagnosis

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=flat-square)](https://huggingface.co)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Thesis](https://img.shields.io/badge/Master's%20Thesis-BAU%202025-blue?style=flat-square)](https://bau.edu.tr)

**Master's Thesis · Bahçeşehir University · Department of Computer Engineering · August 2025**

*Supervisor: Asst. Prof. Fatih Kahraman*

</div>

---

## Abstract

This thesis investigates symptom-based disease classification by rigorously comparing traditional machine learning models with specialized transformer architectures. Data from the **Kaggle Symptom2Disease** and **Gretel Symptom-to-Diagnosis** datasets are harmonized into 22 common disease classes, and the **TecBlic** synthetic dataset (45 classes) is also included. Models are evaluated both in-domain and cross-domain, and robustness to vocabulary shifts, demographic prompts, and input noise is tested.

Linear classifiers (Logistic Regression, SVM, Random Forest) trained on TF–IDF and character n-grams serve as strong baselines **(macro-F1 ≈ 0.96–0.97)**. Domain-specific transformer models (BioBERT, ClinicalBERT) require extensive fine-tuning: a 20-epoch, class-weighted BioBERT run achieved **macro-F1 = 0.982** on the Kaggle test set and **0.970** on a Gretel→Kaggle transfer. On the larger TecBlic corpus, a class-weighted transformer achieved **98.7% accuracy** and **macro-F1 = 0.986**.

Key findings reveal that robust classical baselines remain competitive with transformers, AdamP outperforms AdamW for fine-tuning, and cross-domain generalization is asymmetric — models trained on Gretel generalize better to Kaggle than vice versa (≈6–7% drop).

**Keywords:** Symptom-to-Diagnosis Classification · Clinical NLP · Classical Models · Transformer Models · Domain Adaptation

---

## Visualizations

<div align="center">

**Macro-F1 Comparison — Gretel→Kaggle Transfer**
![Model Comparison](results/macro_f1_gretel_kaggle.png)

**Gretel Dataset — 22-Class Train/Test Distribution**
![Gretel Dataset Split](results/gretel_dataset_split.png)

</div>

---

## Results Summary

### In-Domain Performance (Kaggle & Gretel)

| Model | Kaggle CV Macro-F1 | Gretel Test Macro-F1 | Gretel→Kaggle Transfer |
|---|---|---|---|
| Logistic Regression (TF-IDF) | ~0.96 | ~0.96 | ~0.95 |
| LR + Character N-grams | ~0.97 | ~0.97 | ~0.97 |
| SVM (TF-IDF) | ~0.96 | ~0.96 | ~0.95 |
| Random Forest | ~0.94 | ~0.93 | ~0.91 |
| ClinicalBERT (3 epochs) | ~0.95 | ~0.94 | ~0.93 |
| **BioBERT (20 epochs, class-weighted)** | **0.982** | — | **0.970** |

### TecBlic Dataset (45-class, Synthetic)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Classical Baselines (best) | ~0.95 | ~0.94 |
| **BioBERT (class-weighted)** | **98.7%** | **0.986** |

### Optimizer Comparison (BioBERT Fine-Tuning)

| Optimizer | Kaggle CV Macro-F1 | Gretel→Kaggle Macro-F1 | Training Time |
|---|---|---|---|
| AdamW | ~0.96 | ~0.96 | Baseline |
| **AdamP** | **~0.97** | **~0.97** | +40% |

> **Key finding:** AdamP converges faster and achieves higher macro-F1, at the cost of ~40% additional training time.

### Cross-Domain Transfer (Asymmetry Finding)

| Train → Test | Macro-F1 | Δ vs In-Domain |
|---|---|---|
| Gretel → Kaggle | ~0.97 | Minimal drop |
| Kaggle → Gretel | ~0.90–0.91 | **≈6–7% drop** |

> Models trained on Gretel's conversational style generalize better to structured Kaggle phrases than vice versa, suggesting that richer language diversity in training improves robustness.

---

## Repository Structure

```
Clinical-Decision-Support-NLP/
│
├── src/                              # Reusable Python modules
│   ├── __init__.py
│   ├── preprocessing.py              # Data loading, cleaning, label harmonization
│   ├── classical_models.py           # LR, SVM, RF pipelines + experiment runner
│   └── evaluation.py                 # Metrics: macro-F1, ECE, calibration, AURC
│
├── scripts/                          # Standalone experiment scripts
│   ├── optimizer_comparison.py       # AdamW vs AdamP comparison (Table 9)
│   ├── tecblic_finetuning.py         # BioBERT on TecBlic → 98.7% accuracy
│   └── biobert_experiments.py        # Extended BioBERT framework (all configs)
│
├── results/                          # Output figures and metrics
│   ├── macro_f1_gretel_kaggle.png    # Model comparison chart (Gretel→Kaggle)
│   ├── gretel_dataset_split.png      # Gretel 22-class train/test distribution
│   ├── gretel_class_distribution.png # Full Gretel class distribution
│   └── classical_results.csv         # Saved experiment results
│
├── data/                             # Data directory (CSVs not committed — see below)
│   └── README_data.md
│
├── requirements.txt                  # Python dependencies
└── README.md
```

> **Note on data:** Raw and canonicalized CSV files are not committed to this repo due to size. Download instructions are in the Datasets section below.

---

## Datasets

| Dataset | Source | Samples | Classes | Style |
|---|---|---|---|---|
| Symptom2Disease (Kaggle) | [Kaggle](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) | ~1,200 (1,066 after harmonization) | 24 → 22 | Structured phrases |
| Symptom-to-Diagnosis (Gretel) | [Gretel AI](https://huggingface.co/datasets/gretelai/symptom_to_diagnosis) | ~1,100 | 22 | Conversational narratives |
| Symptom-Based Disease Prediction (TecBlic) | [Kaggle](https://www.kaggle.com) | Large synthetic | 45 | Synthetic descriptions |

**Label harmonization:** The Kaggle dataset's 24 classes were reduced to 22 by removing `Acne` and `Dimorphic Hemorrhoids` to align with the Gretel label space, enabling fair cross-domain evaluation.

---

## Models & Approach

### Classical Baselines
Three classical algorithms were implemented using **scikit-learn**:
- **Logistic Regression** — with TF-IDF (unigram + bigram) and character n-grams (3–5 chars)
- **Support Vector Machine (SVM)** — linear kernel with TF-IDF features
- **Random Forest** — ensemble of decision trees on TF-IDF representations

Character n-grams (3–5 chars) were a key preprocessing choice that substantially improved robustness to misspellings and morphological variation.

### Transformer Models
Two domain-specific BERT variants were fine-tuned:
- **BioBERT v1.2** (`dmis-lab/biobert-v1.1`) — pretrained on biomedical literature
- **ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) — pretrained on MIMIC-III clinical notes

**Extended fine-tuning configuration (best run):**
| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Max sequence length | 256 |
| Learning rate | 5×10⁻⁵ |
| Batch size | 16 |
| Weight decay | 0.01 |
| LR scheduler | Cosine with 10% warmup |
| Loss function | Class-weighted cross-entropy |
| Optimizer | AdamP / AdamW (compared) |
| Validation strategy | Stratified 5-fold CV |

---

## Pipeline Overview

```
Raw Symptom Text
       │
       ▼
┌─────────────────────────────────┐
│  Preprocessing                  │
│  • Lowercase + stop-word removal│
│  • TF-IDF (uni/bigram)          │
│  • Character n-grams (3–5)      │
└────────────┬────────────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐    ┌──────────────────┐
│Classical│    │Transformer       │
│ LR/SVM  │    │ BioBERT          │
│ RF      │    │ ClinicalBERT     │
└────┬────┘    └────────┬─────────┘
     │                  │
     └────────┬──────────┘
              ▼
   ┌──────────────────────┐
   │  Evaluation          │
   │  • Macro-F1          │
   │  • Cross-domain test │
   │  • Robustness probes │
   └──────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Optimizer | AdamP > AdamW | Faster convergence, +1% macro-F1 on transfer tasks |
| N-gram type | Character (3–5) > Word | Handles misspellings, morphological variation |
| Class weighting | Inverse-frequency weights | Addresses label imbalance in fine-tuning |
| Structured features | Not included | Binary symptom flags + age/gender did not improve F1 |
| Evaluation | 5-fold stratified CV | Ensures unbiased, reproducible estimates |
| Label space | 22 harmonized classes | Enables fair Kaggle ↔ Gretel cross-domain comparison |

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Mohammed-jorf/Clinical-Decision-Support-NLP.git
cd Clinical-Decision-Support-NLP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt (core dependencies)
```
torch>=2.0.0
transformers>=4.35.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
adamp
matplotlib
seaborn
jupyter
```

---

## Reproducing Results

### 1. Prepare data

```python
from src.preprocessing import load_and_harmonize

# Expects data/symptom2disease.csv, data/gretel_train.csv, data/gretel_test.csv
kaggle22, gretel_train22, gretel_test22 = load_and_harmonize("data/")
# Saves kaggle_22_canon.csv, gretel_train_22_canon.csv, gretel_test_22_canon.csv
```

### 2. Classical baselines (macro-F1 ≈ 0.96–0.97)

```python
from src.classical_models import run_classical_experiments
results = run_classical_experiments(kaggle22, gretel_train22, gretel_test22)
```

### 3. Optimizer comparison (AdamW vs AdamP)

```bash
python scripts/optimizer_comparison.py \
    --data-dir data/ \
    --model-name dmis-lab/biobert-v1.1 \
    --epochs 20 \
    --output results/optimizer_comparison.csv
```

### 4. TecBlic fine-tuning (98.7% accuracy)

```bash
python scripts/tecblic_finetuning.py \
    --model-name dmis-lab/biobert-base-cased-v1.2 \
    --epochs 5 \
    --output-dir results/tecblic/
```

### 5. Extended BioBERT experiments

```bash
python scripts/biobert_experiments.py \
    --data-dir data/ \
    --experiment kaggle_cv \
    --optimizer adamp \
    --scheduler cosine \
    --epochs 20
```

> ⚠️ **Note:** Transformer fine-tuning requires a GPU. All experiments were run with a CUDA-enabled GPU. CPU-only runs are supported but will be significantly slower.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{jorf2025clinical,
  author    = {Mohammed Ibrahim A. Jorf},
  title     = {Enhancing Clinical Decision Support: Benchmarking Machine Learning
               and Transformer Models for Symptom-Based Diagnosis},
  school    = {Bahçeşehir University},
  year      = {2025},
  month     = {August},
  type      = {Master's Thesis},
  address   = {Istanbul, Turkey},
  advisor   = {Asst. Prof. Fatih Kahraman}
}
```

---

## Author

**Mohammed Ibrahim A. Jorf**
Master's in Computer Engineering — Bahçeşehir University, Istanbul (2025)

[![GitHub](https://img.shields.io/badge/GitHub-Mohammed--jorf-181717?style=flat-square&logo=github)](https://github.com/Mohammed-jorf)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
