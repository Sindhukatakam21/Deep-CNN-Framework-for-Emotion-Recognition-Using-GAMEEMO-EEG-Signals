# 🧠 EEG-Based Emotion Recognition using Deep CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-97.06%25-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-GAMEEMO-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

> A Deep Convolutional Neural Network framework that classifies human emotions from 14-channel EEG signals into **Boring, Calm, Horror, and Happy** — achieving **97.06% accuracy** on the GAMEEMO dataset.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Model Architecture](#-model-architecture)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Inference](#-inference)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)
- [References](#-references)
- [License](#-license)

---

## 🔍 Overview

Emotion recognition from EEG signals is a challenging problem in **affective computing** and **brain-computer interfaces (BCI)**. Traditional machine learning methods struggle with the non-linear, non-stationary nature of brain signals and high inter-subject variability.

This project addresses these challenges using a **Deep CNN** that extracts hierarchical spatiotemporal features from raw EEG signals — outperforming LSTM, Bi-LSTM, and baseline CNN models.

**Key Highlights:**
- ✅ 97.06% classification accuracy
- ✅ 99.24% subject-wise accuracy
- ✅ 4-class emotion classification (Boring / Calm / Horror / Happy)
- ✅ End-to-end pipeline: raw EEG → preprocessed signal → trained model → predicted emotion
- ✅ Faster convergence with GPU acceleration

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Deep CNN (Proposed)** | **97.06%** | **0.97** | **0.97** | **0.97** |
| LSTM | 91.18% | 0.91 | 0.91 | 0.91 |
| CNN (Baseline) | 89.92% | 0.90 | 0.89 | 0.89 |
| Bi-LSTM | 87.39% | 0.87 | 0.87 | 0.87 |

> 📁 Confusion matrix, accuracy/loss curves, and comparison plots are available in the [`results/`](./results/) folder.

---

## 📂 Dataset

This project uses the **GAMEEMO** dataset — a publicly available EEG dataset designed for game-induced emotion recognition.

| Property | Details |
|----------|---------|
| EEG Channels | 14 |
| Emotion Classes | Boring, Calm, Horror, Happy |
| Stimuli | Game scenarios designed to elicit distinct emotions |
| Sampling Rate | 128 Hz |

**Dataset Split used in this project:**

| Split | Proportion |
|-------|-----------|
| Training | 68% |
| Validation | 12% |
| Testing | 20% |

> 📥 Download the GAMEEMO dataset from the [official source](https://doi.org/10.1016/j.bspc.2020.102095) and place it in the `data/raw/` directory.

---

## 📁 Project Structure

```
EEG-Emotion-Recognition/
│
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
├── LICENSE
│
├── data/
│   ├── raw/                         ← Place GAMEEMO dataset here
│   └── processed/                   ← Preprocessed segments (auto-generated)
│
├── src/
│   ├── preprocess.py                ← Filtering, normalization, segmentation
│   ├── model.py                     ← Deep CNN architecture
│   ├── train.py                     ← Training loop
│   ├── evaluate.py                  ← Metrics and evaluation
│   └── predict.py                   ← Single-sample inference
│
├── notebooks/
│   └── EEG_Emotion_CNN.ipynb        ← Full end-to-end walkthrough
│
├── models/
│   └── best_model.pth               ← Saved best model weights
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_loss_curve.png
│   └── model_comparison.png
│
└── docs/
    └── Project_Report.pdf           ← Full project report
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (recommended for training)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/EEG-Emotion-Recognition.git
cd EEG-Emotion-Recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the GAMEEMO dataset and place it in `data/raw/`. Then run preprocessing:

```bash
python src/preprocess.py --input data/raw/ --output data/processed/
```

### 4. Train the Model

```bash
python src/train.py --data data/processed/ --epochs 50 --batch_size 32
```

### 5. Evaluate

```bash
python src/evaluate.py --model models/best_model.pth --data data/processed/
```

### 6. Predict on a Single Sample

```bash
python src/predict.py --model models/best_model.pth --input sample_eeg.csv
```

---

## 🏗️ Model Architecture

The Deep CNN consists of **4 progressive convolutional blocks** followed by Global Average Pooling and a fully connected classification head.

```
Input (14-channel EEG segment)
        │
        ▼
┌─────────────────────────────┐
│  Block 1: Conv1D (64, k=7)  │
│  BatchNorm → ReLU → MaxPool │
│  Dropout (0.25)             │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Block 2: Conv1D (128, k=5) │
│  BatchNorm → ReLU → MaxPool │
│  Dropout (0.25)             │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Block 3: Conv1D (256, k=3) │
│  BatchNorm → ReLU → MaxPool │
│  Dropout (0.30)             │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Block 4: Conv1D (512, k=3) │
│  BatchNorm → ReLU → MaxPool │
│  Dropout (0.40)             │
└─────────────┬───────────────┘
              ▼
   Global Average Pooling
              ▼
     Dense → Softmax (4)
              ▼
  Output: Emotion Class
```

---

## ⚙️ Preprocessing Pipeline

Raw EEG signals go through three stages before being fed to the model:

1. **Band-pass Filtering (0.5 – 45 Hz)**
   Removes low-frequency drift, high-frequency noise, and muscle artifacts using a Butterworth filter.

2. **Z-score Normalization**
   Standardizes signal amplitude across all 14 channels to zero mean and unit variance, reducing inter-subject variability.

3. **Sliding Window Segmentation (50% overlap)**
   Divides continuous EEG recordings into fixed-length windows for training, effectively augmenting dataset size and improving temporal learning.

---

## 🏋️ Training

```bash
python src/train.py \
  --data data/processed/ \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001 \
  --output models/best_model.pth
```

**Training Configuration:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 50 |

Training logs and loss/accuracy curves are saved automatically to the `results/` directory.

---

## 📈 Evaluation

```bash
python src/evaluate.py \
  --model models/best_model.pth \
  --data data/processed/
```

Outputs:
- Overall accuracy, precision, recall, F1-score
- Per-class classification report
- Confusion matrix (saved to `results/confusion_matrix.png`)

---

## 🔮 Inference

Run prediction on a single EEG sample:

```bash
python src/predict.py --model models/best_model.pth --input sample_eeg.csv
```

**Sample output:**
```
Loading model... ✓
Preprocessing input... ✓
Running inference...

Predicted Emotion : CALM
Confidence        : 98.3%
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch |
| Signal Processing | SciPy, NumPy |
| Data Handling | Pandas |
| Visualization | Matplotlib, Plotly |
| Notebook | Jupyter |
| Hardware | GPU (CUDA) |

---

## 🔭 Future Work

- [ ] Real-time emotion detection with live EEG streaming
- [ ] Integration with portable wearable EEG devices
- [ ] Transformer-based architectures (EEGFormer, EEG-Conformer)
- [ ] Cross-dataset generalization and validation
- [ ] Personalized models via transfer learning
- [ ] Deployment as a REST API for healthcare applications

---

## 📚 References

1. P. Sreehari et al. (2025). EEG-based Emotion Recognition for Affective Computing Applications.
2. T. Alakus and I. Turkoglu (2020). Comparison of Deep Learning Approaches for EEG-Based Emotion Recognition. *Procedia Computer Science.*
3. A. Abgeena and S. Garg (2025). Deep Learning Models for EEG Signal Classification in Emotion Analysis.
4. J. Vakala Rani et al. (2025). Spatiotemporal CNN Architectures for Multi-Channel EEG Processing.
5. K. Shahzad et al. (2024). Robust EEG Emotion Recognition with Convolutional Feature Extraction.
6. M. Mansour et al. (2025). Affective State Detection Using Deep Neural Networks on EEG Signals.
7. A. Ibrahim et al. (2026). Next-Generation Brain-Computer Interfaces for Emotion-Aware Systems.
8. GAMEEMO Dataset (2020). A Game-Based EEG Dataset for Emotion Recognition Research.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🙋 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
- Email: your.email@example.com

---

> ⭐ If you found this project useful, please consider giving it a star — it helps others discover it!
