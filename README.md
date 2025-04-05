# SU_PA2
# Speaker Verification & Language Identification using HuBERT and Audio Features

This repository contains two Jupyter Notebooks implementing:
1. **Speaker Verification** using the HuBERT model and the VoxCeleb1 dataset.
2. **Language Identification** on speech samples, with a focus on Urdu.

---

## 🔹 Notebooks

### `Ques_1.ipynb`: Speaker Verification

- Uses [HuBERT (facebook/hubert-large-ls960-ft)](https://huggingface.co/facebook/hubert-large-ls960-ft) model.
- Evaluates speaker pairs using the VoxCeleb1 cleaned trial list.
- Computes cosine similarity between speaker embeddings.
- Outputs verification accuracy and ROC metrics.

#### Requirements:
- `transformers`
- `torchaudio`
- `librosa`
- `scikit-learn`
- `tqdm`

### `Ques_2.ipynb`: Language Identification

- Processes and visualizes speech signals.
- Identifies spoken language (example: Urdu).
- Uses audio feature extraction (MFCCs/spectrograms).
- Plots language classification results and intermediate features.

#### Requirements:
- `matplotlib`
- `numpy`
- `librosa`
- `scikit-learn`

---

## 🛠 Setup Instructions

```bash
# Create a virtual environment (optional)
python -m venv env
source env/bin/activate  # or .\env\Scripts\activate on Windows

# Install dependencies
pip install torch torchaudio transformers librosa scikit-learn matplotlib tqdm
