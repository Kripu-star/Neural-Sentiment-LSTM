# 🎭 NeuralSentiment: Bidirectional LSTM Sentiment Engine

A high-performance, end-to-end Deep Learning application that performs binary sentiment classification on 50,000+ text sequences using **Bidirectional LSTMs**. Includes a real-time inference dashboard built with **Streamlit**.

## 🚀 Project Overview
This project was developed to explore the nuances of **Sequential Data Processing** and the mathematical foundations of **Recurrent Neural Networks (RNNs)**. While modern Transformers are the standard, this implementation focuses on the efficiency and memory-cell mechanics of LSTMs for real-time text analysis.

## ✨ Key Features
* **Architecture:** Many-to-one Bidirectional LSTM with optimized Dropout layers (0.3).
* **High-Speed Inference:** Real-time prediction with <50ms latency per sequence.
* **Streamlit Dashboard:** Interactive Web UI for single-text and batch-mode analysis.
* **Metric Visualization:** Real-time plotting of training loss and accuracy curves.

## 🏗️ Technical Implementation
### Model Architecture
```python
LSTMSentimentClassifier(
  (embedding): Embedding(10000, 128)
  (lstm): LSTM(128, 128, bidirectional=True, dropout=0.3)
  (dropout): Dropout(p=0.5)
  (fc): Linear(256, 2)
)