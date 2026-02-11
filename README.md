# Clinical-Decision-Support-NLP
# Enhancing Clinical Decision Support: NLP Benchmarking 🏥

This repository contains the official implementation of my MSc Thesis: **"Enhancing Clinical Decision Support Systems through NLP."**

The project benchmarks **Classical Machine Learning** against **Domain-Specific Transformers (BioBERT, ClinicalBERT)** for symptom-based disease diagnosis.

### 📊 Key Results
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **BioBERT (Fine-Tuned)** | **98.7%** | **0.98** | **0.99** | **0.98** |
| ClinicalBERT | 97.4% | 0.97 | 0.97 | 0.97 |
| Random Forest | 94.2% | 0.94 | 0.93 | 0.94 |
| SVM | 92.1% | 0.92 | 0.91 | 0.92 |

### 🔬 Methodology
1.  **Datasets:** Aggregated datasets including **Symptom2Disease** (Kaggle) and synthetic medical records.
2.  **Preprocessing:** Compared TF-IDF/N-Gram approaches (for classical ML) vs. BERT Tokenization.
3.  **Optimization:** Implemented **Class-Weighted Loss** to handle dataset imbalance and used **AdamW** for stable transformer convergence.

### 📈 Visualizations
#### Confusion Matrix (BioBERT)
![Confusion Matrix](/Users/mohammedjorf/Downloads/confusion_matrix_biobert-Copy1.png)

#### Training Loss Convergence
![Loss Curve](/Users/mohammedjorf/Downloads/accuracy_comparison.png)

### 🛠️ Installation & Usage
```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_BioBERT_Finetuning.ipynb
