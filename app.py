#!/usr/bin/env python3
"""
🎭 LSTM Sentiment Analysis Streamlit Application
Optimized for high-performance neural inference
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import re
from datetime import datetime
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="🎭 Neural Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .positive-sentiment {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .negative-sentiment {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# LSTM Model Architecture (Must match train_simple.py)
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMSentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes) 
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(output)
        return self.classifier(output)

@st.cache_resource
def load_model_and_vocab():
    """Load Model and Vocabulary with Caching"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        vocab_path = Path("vocabulary.pkl")
        if not vocab_path.exists():
            st.error("❌ vocabulary.pkl not found! Run train_simple.py first.")
            return None, None, None
            
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        model = LSTMSentimentClassifier(vocab_size=len(vocab))
        model_path = Path("best_lstm_sentiment_model.pth")
        
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            st.success("✅ Neural Engine Loaded Successfully!")
        else:
            st.warning("⚠️ Trained model weights not found.")
            return None, None, None
            
        model.to(device)
        model.eval()
        return model, vocab, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

def predict_sentiment(text, model, vocab, device, max_length=256):
    """Perform Inference on input text"""
    if model is None: return "Unknown", 0.0, [0.5, 0.5]
    model.eval()
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(indices) < max_length:
        indices.extend([vocab['<PAD>']] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    input_ids = torch.tensor([indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        confidence, predicted = torch.max(probabilities, 1)
        
    sentiment = "Positive" if predicted.item() == 1 else "Negative"
    return sentiment, confidence.item(), probabilities[0].cpu().numpy()

def analyze_text_features(text):
    """Analyze Quantitative Text Metrics"""
    features = {
        'Character Count': len(text),
        'Word Count': len(text.split()),
        'Sentence Count': max(1, len(re.split(r'[.!?]+', text)) - 1),
        'Uppercase Ratio (%)': sum(1 for c in text if c.isupper()) / len(text) * 100 if text else 0,
        'Punctuation Count': sum(1 for c in text if c in '.,!?;:'),
        'Avg. Word Length': np.mean([len(word) for word in text.split()]) if text.split() else 0
    }
    return features

def main():
    st.markdown('<h1 class="main-header">🎭 Neural Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar Section
    st.sidebar.markdown("## 🛠️ Configuration")
    sidebar_tab = st.sidebar.radio(
        "Navigation:",
        ["📊 Model Stats", "ℹ️ About Engine", "📖 User Guide"]
    )
    
    with st.spinner("🔄 Initializing Neural Engine..."):
        model, vocab, device = load_model_and_vocab()
    
    if sidebar_tab == "📊 Model Stats":
        st.sidebar.markdown("### 📈 System Metrics")
        if model and vocab:
            st.sidebar.info(f"""
            **Architecture:** Bidirectional LSTM
            **Vocab Size:** {len(vocab):,}
            **Inference Device:** {device.type.upper()}
            **Sequence Limit:** 256 tokens
            """)
            
    elif sidebar_tab == "ℹ️ About Engine":
        st.sidebar.markdown("### ℹ️ Technical Overview")
        st.sidebar.markdown("""
        **Neural Sentiment Platform**
        
        This system utilizes Deep Learning (LSTM) to detect emotional variance in textual data.
        
        **🔧 Tech Stack:**
        - **Model:** PyTorch LSTM
        - **Dataset:** IMDB 50K
        - **Performance:** 87.2% Accuracy
        """)
        
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Text Input")
        input_method = st.radio(
            "Select input source:",
            ["✍️ Manual Entry", "📋 Example Templates", "📁 File Upload"]
        )
        
        text_input = ""
        if input_method == "✍️ Manual Entry":
            text_input = st.text_area("Input text for sentiment analysis:", height=150, placeholder="The cinematography was brilliant, but the pacing felt slow...")
            
        elif input_method == "📋 Example Templates":
            examples = {
                "Positive Review": "This movie is absolutely amazing! The acting was superb and the plot kept me engaged.",
                "Negative Review": "Terrible movie, complete waste of time. Poor acting and boring scenes.",
                "Mixed Sentiment": "It was okay, nothing special but not bad either. Average movie with decent acting."
            }
            selected_example = st.selectbox("Choose a template:", list(examples.keys()))
            text_input = examples[selected_example]
            st.text_area("Template Content:", value=text_input, height=100, disabled=True)

        elif input_method == "📁 File Upload":
            uploaded_file = st.file_uploader("Upload .txt or .csv", type=['txt', 'csv'])
            if uploaded_file:
                text_input = str(uploaded_file.read(), "utf-8")
        
        if st.button("🔍 Run Neural Analysis", type="primary", use_container_width=True):
            if text_input.strip():
                with st.spinner("🤖 Processing sequence..."):
                    sentiment, confidence, probabilities = predict_sentiment(text_input, model, vocab, device)
                    
                    st.markdown("### 🎯 Inference Results")
                    if sentiment == "Positive":
                        st.markdown(f'<div class="positive-sentiment"><h3>😊 Positive Sentiment</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="negative-sentiment"><h3>😞 Negative Sentiment</h3><p>Confidence: {confidence:.2%}</p></div>', unsafe_allow_html=True)
                    
                    # Probability Distribution Chart
                    fig = go.Figure(data=[go.Bar(x=['Negative', 'Positive'], y=probabilities, marker_color=['#ff6b6b', '#51cf66'], text=[f'{p:.2%}' for p in probabilities], textposition='auto')])
                    fig.update_layout(title="Probability Distribution", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please provide text for analysis.")
    
    with col2:
        st.markdown("### 📈 Linguistic Features")
        if text_input.strip():
            features = analyze_text_features(text_input)
            for f, v in features.items():
                st.metric(f, f"{v:.1f}" if isinstance(v, float) else v)
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'><p>🚀 LSTM Sentiment Analysis Engine | Optimized for PyTorch | Developed by Pushpam</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()