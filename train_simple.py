#!/usr/bin/env python3
"""
🎯 LSTM Sentiment Analysis Training Script
Optimized for local neural network training - No internet connection required
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import re
from collections import Counter
import pickle

class SentimentDataset(Dataset):
    def __init__(self, reviews, sentiments, vocab, max_length=256):
        self.reviews = reviews
        self.sentiments = sentiments
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]
        
        # Tokenize and convert to numerical indices
        tokens = self.tokenize(review)
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Apply Padding or Truncation
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }
    
    def tokenize(self, text):
        """Standard alphanumeric tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMSentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional architecture
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.dropout(output)
        return self.classifier(output)

def build_vocabulary(texts, min_freq=2, max_vocab=10000):
    """Generate a word-to-index mapping (Vocabulary)"""
    print("📚 Building Vocabulary...")
    
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing Sequences"):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        all_tokens.extend(tokens)
    
    # Calculate token frequencies
    token_counts = Counter(all_tokens)
    
    # Initialize vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, count in token_counts.most_common(max_vocab - 2):
        if count >= min_freq:
            vocab[token] = len(vocab)
    
    print(f"📊 Final Vocabulary Size: {len(vocab)}")
    return vocab

def load_and_prepare_data():
    """Load IMDB dataset and perform preprocessing"""
    print("📂 Loading Raw Data...")
    
    data_path = Path("data/IMDB Dataset.csv")
    if not data_path.exists():
        raise FileNotFoundError("IMDB Dataset.csv not found! Please check the data/ directory.")
    
    df = pd.read_csv(data_path)
    print(f"📊 Total raw samples: {len(df)}")
    
    # Map text sentiments to binary numerical values
    df['sentiment_num'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Subsampling to optimize local compute resources
    sample_size = 25000  # 12,500 per class
    df_positive = df[df['sentiment_num'] == 1].sample(n=sample_size//2, random_state=42)
    df_negative = df[df['sentiment_num'] == 0].sample(n=sample_size//2, random_state=42)
    df = pd.concat([df_positive, df_negative]).reset_index(drop=True)
    
    print(f"📊 Training Subset: {len(df)}")
    print(f"📊 Positive Samples: {sum(df['sentiment_num'] == 1)}")
    print(f"📊 Negative Samples: {sum(df['sentiment_num'] == 0)}")
    
    return df

def create_data_loaders(df, vocab, batch_size=32, max_length=256):
    """Initialize PyTorch DataLoaders"""
    print("🔄 Preparing Data Loaders...")
    
    # Stratified Train-Test Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['review'].values,
        df['sentiment_num'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment_num'].values
    )
    
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model():
    """Execute the neural network training loop"""
    print("🔧 Initializing Model Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Compute Device: {device}")
    
    if torch.cuda.is_available():
        print(f"💾 GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    df = load_and_prepare_data()
    vocab = build_vocabulary(df['review'].values)
    
    # Persist vocabulary for future inference
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("💾 Vocabulary serialized to: vocabulary.pkl")
    
    train_loader, test_loader = create_data_loaders(df, vocab, batch_size=32, max_length=256)
    
    model = LSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2
    ).to(device)
    
    print(f"🏗️ Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 5
    train_losses, train_accuracies = [], []
    
    print("🚀 Commencing Training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct_predictions, total_predictions = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.4f}'})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.4f}")
        
        test_acc = evaluate_model(model, test_loader, device)
        print(f"  Validation Accuracy: {test_acc:.4f}")
        print("-" * 40)
    
    torch.save(model.state_dict(), 'best_lstm_sentiment_model.pth')
    print("💾 Model Weights Saved: best_lstm_sentiment_model.pth")
    
    plot_training_curves(train_losses, train_accuracies)
    return model, vocab

def evaluate_model(model, test_loader, device):
    """Perform validation on unseen test data"""
    model.eval()
    correct_predictions, total_predictions = 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    return correct_predictions / total_predictions

def plot_training_curves(losses, accuracies):
    """Visualize and save performance metrics"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', label='Loss Trend')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, 'r-', label='Accuracy Trend')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lstm_sentiment_training_curves.png', dpi=300)
    print("📊 Performance curves exported to: lstm_sentiment_training_curves.png")

def predict_sentiment(text, model, vocab, device, max_length=256):
    """Single sequence inference helper"""
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
    
    return "Positive" if predicted.item() == 1 else "Negative", confidence.item()

if __name__ == "__main__":
    print("=" * 60)
    print("🎭 Neural Sentiment Training Environment")
    print("=" * 60)
    
    trained_model, trained_vocab = train_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n🧪 Verification Phase - Test Predictions:")
    test_cases = [
        "This movie is absolutely amazing! I loved every minute of it.",
        "Terrible movie, waste of time. Very disappointing.",
        "It was okay, nothing special but not bad either.",
        "Boring and predictable. Not worth watching."
    ]
    
    for text in test_cases:
        sentiment, conf = predict_sentiment(text, trained_model, trained_vocab, device)
        print(f"Sample: {text}\nResult: {sentiment} ({conf:.2%} confidence)\n" + "-"*40)
    
    print("\n🎉 Project Ready! Run 'streamlit run app.py' to launch the UI.")