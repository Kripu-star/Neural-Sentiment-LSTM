# 🎭 NeuralSentiment: Bidirectional LSTM Sentiment Engine

A high-performance, end-to-end Deep Learning application that performs binary sentiment classification on 50,000+ text sequences using **Bidirectional LSTMs**. Includes a real-time inference dashboard built with **Streamlit**.

## 🚀 Project Overview
This project was developed to explore the nuances of **Sequential Data Processing** and the mathematical foundations of **Recurrent Neural Networks (RNNs)**. While modern Transformers are the standard, this implementation focuses on the efficiency and memory-cell mechanics of LSTMs for real-time text analysis.

## ✨ Key Features
* **Architecture:** Many-to-one Bidirectional LSTM with optimized Dropout layers (0.3).
* **High-Speed Inference:** Real-time prediction with <50ms latency per sequence.
* **Streamlit Dashboard:** Interactive Web UI for single-text and batch-mode analysis.
* **Metric Visualization:** Real-time plotting of training loss and accuracy curves.
## 📊 Performance Analysis & Inference
The dashboard provides real-time probability distributions for text sequences. Below are three distinct cases processed by the neural engine:

#### Case 1: High-Confidence Positive Sentiment
![Positive Case](./assets/dashboard_pos.png)
*The model extracts strong semantic features from high-value tokens like 'amazing' and 'brilliant', resulting in a confidence score $>90\%$.*

#### Case 2: Mixed Sentiment (Linguistic Nuance)
![Mixed Case](./assets/mixed_sentiment.png)
*Analyzing the sequence: "The cinematography was amazing but the acting was low." The model correctly identifies the conflict introduced by the coordinating conjunction 'but', pulling the negative probability to $~37.1\%$.*

#### Case 3: Clear Negative Sentiment
![Negative Case](./assets/dashboard_neg.png)
*Demonstrates robust detection of negative polarity in reviews with high emotional magnitude words like 'terrible' or 'disappointing'.*
## 🏗️ Technical Implementation
### Model Architecture
    ```python 
             LSTMSentimentClassifier(
            (embedding): Embedding(10000, 128)
            (lstm): LSTM(128, 128, bidirectional=True, dropout=0.3)
             (dropout): Dropout(p=0.5)
             (fc): Linear(256, 2)
              )
## 📈 Training Results & Convergence
To ensure the model avoids the overfitting trap common in deep recurrent networks, I implemented **Stratified Subsampling** and **Dropout Regularization (0.3)**.

![Training Curves](./assets/curves.png)
*Fig 1: Categorical Cross-Entropy Loss and Accuracy convergence over 5 epochs. The smooth decline in loss proves stable weight updates during the Adam optimization process.*

## 💻 Local Execution Guide
To run this dashboard on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/Neural-Sentiment-LSTM.git](https://github.com/your-username/Neural-Sentiment-LSTM.git)
   cd Neural-Sentiment-LSTM
   python -m venv venv
   source venv/bin/activate # On Windows use: .\venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
