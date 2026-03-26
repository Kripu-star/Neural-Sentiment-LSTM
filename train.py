#!/usr/bin/env python3
"""
🎯 BERT Sentiment Analysis Training Script
Optimized for RTX 5060 8GB - High Performance Transformer Training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_length=256):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]
        
        # Tokenize and encode the text
        encoding = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

class BERTSentimentClassifier(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout=0.3):
        super(BERTSentimentClassifier, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained(model_name, local_files_only=False)
        except:
            # Fallback if internet download fails
            print(f"⚠️ {model_name} download failed, attempting to use local model...")
            from transformers import DistilBertModel, DistilBertConfig
            config = DistilBertConfig()
            self.bert = DistilBertModel(config)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Memory Optimization: Freeze the base BERT embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Fine-tuning: Only train the final 2 layers of the encoder
        for layer in self.bert.encoder.layer[:-2]:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Standard BERT practice: Use the [CLS] token (index 0) representation
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.dropout(pooled_output)
        output = self.classifier(output)
        
        return output

def load_and_prepare_data():
    """Load and preprocess the dataset"""
    print("📊 Loading IMDB dataset...")
    
    # Load raw CSV
    df = pd.read_csv('./data/IMDB Dataset.csv')
    
    # Map text labels to numerical values
    df['sentiment_num'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Subsampling for local compute (Optimized for RTX 5060 8GB)
    df = df.sample(n=25000, random_state=42).reset_index(drop=True)
    
    print(f"📊 Training Subset Size: {len(df)}")
    print(f"📊 Positive Samples: {len(df[df['sentiment'] == 'positive'])}")
    print(f"📊 Negative Samples: {len(df[df['sentiment'] == 'negative'])}")
    
    return df

def create_data_loaders(df, tokenizer, batch_size=16, max_length=256):
    """Initialize PyTorch DataLoaders"""
    
    # Stratified Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'].values,
        df['sentiment_num'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment_num'].values
    )
    
    # Create Dataset objects
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, max_length)
    
    # Data loaders (Batch size of 16 is optimal for 8GB VRAM)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model():
    """Execute BERT fine-tuning"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Compute Device: {device}")
    
    if torch.cuda.is_available():
        print(f"💾 GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load dataset
    df = load_and_prepare_data()
    
    # Initialize Tokenizer (DistilBERT is a lightweight transformer)
    model_name = 'distilbert-base-uncased'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    except:
        print("⚠️ Tokenizer download failed, checking local directory...")
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(model_name, local_files_only=True)
    
    # Prepare data loaders
    train_loader, test_loader = create_data_loaders(
        df, tokenizer, batch_size=16, max_length=256
    )
    
    # Initialize Neural Engine
    model = BERTSentimentClassifier(model_name=model_name).to(device)
    
    # Optimizer and Scheduler (Standard for Transformers)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    num_epochs = 3 # BERT converges much faster than LSTMs
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accuracies = []
    
    print(f"\n🎯 Training Started - Running {num_epochs} Epochs")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct_predictions, total_predictions = 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Real-time progress update
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1} Summary:')
        print(f'  Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.4f}')
        
        val_accuracy = evaluate_model(model, test_loader, device)
        print(f'  Validation Accuracy: {val_accuracy:.4f}')
        print('-' * 40)
    
    final_accuracy = evaluate_model(model, test_loader, device)
    print(f'\n🎉 Final Evaluation Accuracy: {final_accuracy:.4f}')
    
    # Persist artifacts
    torch.save(model.state_dict(), 'bert_sentiment_model.pth')
    tokenizer.save_pretrained('./bert_tokenizer')
    print('✅ Neural Engine and Tokenizer successfully serialized!')
    
    plot_training_curves(train_losses, train_accuracies)
    return model, tokenizer

def evaluate_model(model, test_loader, device):
    """Validation performance check"""
    model.eval()
    correct_predictions, total_predictions = 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    
    return correct_predictions / total_predictions

def plot_training_curves(losses, accuracies):
    """Generate and save performance visualizations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss Trend')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(accuracies)
    ax2.set_title('Training Accuracy Trend')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('bert_training_curves.png', dpi=300)
    print('📊 Performance curves exported: bert_training_curves.png')

def predict_sentiment(text, model, tokenizer, device):
    """Single sequence inference helper"""
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        probability = torch.softmax(outputs, dim=1)
    
    sentiment = 'Positive' if predicted.item() == 1 else 'Negative'
    confidence = probability[0][predicted.item()].item()
    
    return sentiment, confidence

if __name__ == "__main__":
    print("=" * 60)
    print("🎭 Transformer-Based Sentiment Analysis (BERT)")
    print("=" * 60)
    
    model, tokenizer = train_model()
    
    print("\n🧪 Inference Test Phase:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_cases = [
        "This movie is absolutely amazing! I loved every minute of it.",
        "Terrible movie, waste of time. Very disappointing.",
        "It was okay, nothing special but not bad either."
    ]
    
    for text in test_cases:
        sentiment, confidence = predict_sentiment(text, model, tokenizer, device)
        print(f"Sample: {text}")
        print(f"Result: {sentiment} ({confidence:.1%} confidence)")
        print("-" * 40)
