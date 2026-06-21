# 📘 Text Generation using RNN, LSTM & GRU

A Deep Learning project for students and beginners to understand how sequence models learn grammar, sentence flow, contextual dependencies, and next-word prediction.

---

## 🎯 Objective

Compare **Vanilla RNN**, **LSTM**, and **GRU** on the same text corpus to understand:
- How each model learns sequential patterns
- Why gated architectures (LSTM/GRU) outperform simple RNNs
- How to generate text using trained models

---

## 🗂️ Project Structure

```
├── Text_Generation_RNN_LSTM_GRU.ipynb   # Main notebook
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook Text_Generation_RNN_LSTM_GRU.ipynb
```

Or open directly in **Google Colab** by uploading the `.ipynb` file.

---

## 🧠 Models Implemented

| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| Vanilla RNN | Embedding → SimpleRNN → Dense | Baseline, prone to vanishing gradients |
| LSTM | Embedding → LSTM → Dense | Input/Forget/Output gates for long-term memory |
| GRU | Embedding → GRU → Dense | Reset/Update gates, faster than LSTM |

---

## 📊 What You'll Learn

- Text tokenization and n-gram sequence creation
- Padding sequences for uniform input length
- Building and training RNN, LSTM, GRU models with Keras
- Comparing training loss across models
- Generating text using a trained model (next-word prediction loop)

---

## ✅ Student Tasks

- Replace the corpus with your own text (song lyrics, story, etc.)
- Increase embedding dimensions and hidden units
- Train for more epochs and observe loss curves
- Generate longer sequences (10–15 words)
- Try stacking two LSTM/GRU layers

---

## 📦 Requirements

See `requirements.txt` for all dependencies. Main libraries used:

- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 📌 Results Summary

- **Vanilla RNN** — Learns short patterns but struggles with long-term dependencies
- **LSTM** — Better at capturing long-range grammar structure
- **GRU** — Similar accuracy to LSTM with faster training time

---

## 🙌 Acknowledgements

Built as a learning project to demonstrate sequence modeling concepts in Deep Learning.
