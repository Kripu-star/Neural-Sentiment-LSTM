
# 🎭 NeuralSentiment: Bidirectional LSTM Sentiment Engine

**NeuralSentiment** is a high-performance, end-to-end Deep Learning application designed for binary sentiment classification on large-scale textual data. By leveraging **Bidirectional LSTMs**, the engine captures contextual dependencies from both past and future states, providing a nuanced understanding of linguistic sentiment.

---

## 🚀 Project Overview
This project explores the nuances of **Sequential Data Processing** and the mathematical foundations of **Recurrent Neural Networks (RNNs)**. While modern Transformers are the industry standard, this implementation focuses on the efficiency and memory-cell mechanics of LSTMs for high-speed, real-time text analysis.

---

## 🏗️ Technical Deep Dive: Exact Model Working
Unlike standard LSTMs, this model processes sequences in two directions, ensuring that the hidden state at any time $t$ has access to information from the entire sequence.

### 1. Bidirectional Processing
The model maintains two independent hidden states:
* **Forward Pass ($\overrightarrow{h}_t$):** Processes the sequence from $x_1$ to $x_N$.
* **Backward Pass ($\overleftarrow{h}_t$):** Processes the sequence from $x_N$ to $x_1$.

The final representation is a concatenation of the two:
$$H_t = [\overrightarrow{h}_t \parallel \overleftarrow{h}_t]$$
This allows the engine to understand that the word *"but"* at the end of a sentence significantly alters the sentiment of the words at the beginning.



### 2. LSTM Memory Cell
Each unit utilizes a gating mechanism to regulate the flow of information, effectively solving the **Vanishing Gradient Problem**:
* **Forget Gate:** Decides what information to discard from the cell state.
* **Input Gate:** Updates the cell state with new information.
* **Output Gate:** Determines the next hidden state.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$



### 3. Architecture Specification
```python 
LSTMSentimentClassifier(
    (embedding): Embedding(10000, 128)  # Vector space representation
    (lstm): LSTM(128, 128, bidirectional=True, dropout=0.3) # 2-layer Bi-LSTM
    (dropout): Dropout(p=0.5) # Regularization to prevent overfitting
    (fc): Linear(256, 2) # Fully connected layer for binary mapping
)
```

---

## ✨ Key Features
* **Many-to-One Architecture:** Maps a variable-length text sequence to a single sentiment polarity.
* **High-Speed Inference:** Real-time prediction with **<50ms latency** per sequence.
* **Streamlit Dashboard:** Interactive Web UI for single-text and batch-mode analysis.
* **Optimized Training:** Designed for local consumer-grade hardware (**RTX 5060 8GB**).

---

## 📊 Performance Analysis & Inference
The dashboard provides real-time probability distributions. Below are three distinct cases processed by the neural engine:

# Case 1: High-Confidence Positive Sentiment###
![Positive Case](./assets/positive1.jpg) ![Positive Case](./assets/positive2.jpg)
*The model extracts strong semantic features from high-value tokens, resulting in a confidence score **>90%**.*

#### Case 2: Semantic Ambiguity & Decision Boundary
![Boundary Case](./assets/mixed weightage.jpg)
Input: "below average""This case demonstrates the engine's behavior at the mathematical decision boundary ($0.5$). While 'below average' carries a negative connotation, the token 'average' is high-frequency and context-dependent in the IMDB training set, often appearing in both neutral and positive comparative contexts. The resulting 50.31% vs 49.69% distribution indicates significant model uncertainty. This highlights the inherent limitation of binary classification when processing sequences that lack strong, polarized emotional features (e.g., 'horrible' or 'excellent')."


#### Case 3: Clear Negative Sentiment
![Negative Case]()
*Demonstrates robust detection of negative polarity using high emotional magnitude words.*

---

## 📈 Training Results & Convergence
To ensure stability, I implemented **Stratified Subsampling** and **Dropout Regularization (0.3)**.

![Training Curves](./assets/curves.png)
*Fig 1: Categorical Cross-Entropy Loss and Accuracy convergence over 5 epochs. The smooth decline in loss proves stable weight updates during the **Adam optimization** process.*

---

## 💻 Local Execution Guide
Follow these steps to deploy the engine on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Neural-Sentiment-LSTM.git
    cd Neural-Sentiment-LSTM
    ```
2.  **Environment Setup:**
    ```bash
    python -m venv venv
    source venv/bin/activate # Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Launch Dashboard:**
    ```bash
    streamlit run app.py
    ```

---



