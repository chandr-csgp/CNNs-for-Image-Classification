# 📈 Stock Price Forecasting — LSTM Time-Series Model

**Sequence-based LSTM neural network for financial time-series prediction (87% accuracy)**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

---

## 📌 Overview

This project applies **Long Short-Term Memory (LSTM)** networks to stock price forecasting. LSTMs are designed to capture long-range dependencies in sequential data — making them well-suited for non-stationary financial time series.

The model uses a **sliding-window approach** to transform raw price data into supervised learning sequences, then trains an LSTM to predict future price movements.

---

## 🏗️ Approach

1. **Data preprocessing** — normalisation, sliding-window sequence construction
2. **Model** — LSTM with configurable look-back window
3. **Training** — Adam optimiser, early stopping to prevent overfitting
4. **Evaluation** — Accuracy score and loss curves

---

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | **87%** |
| Architecture | LSTM (sequence model) |
| Preprocessing | Sliding window + normalisation |

---

## 🔧 Tech Stack

`Python` · `TensorFlow` · `Keras` · `NumPy` · `pandas` · `Matplotlib`

---

## 👤 Author

**Chandra Sekar Putta** — University of Adelaide, MDS 2024–2026  
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
