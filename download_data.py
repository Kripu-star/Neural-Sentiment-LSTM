#!/usr/bin/env python3
"""
IMDB Movie Reviews Sentiment Analysis Dataset Downloader
Optimized for local training environment
"""

import os
import kaggle
import pandas as pd
import torch
from pathlib import Path
from datasets import load_dataset

def download_imdb_dataset():
    """Download the IMDB movie reviews dataset"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("🔄 Downloading IMDB Movie Reviews dataset...")
    
    try:
        # Attempt download from Kaggle
        kaggle.api.dataset_download_files(
            'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
            path='./data',
            unzip=True
        )
        print("✅ IMDB dataset successfully downloaded from Kaggle!")
        
        # Load the CSV file
        df = pd.read_csv('./data/IMDB Dataset.csv')
        
    except Exception as e:
        print(f"⚠️ Kaggle download failed: {e}")
        print("🔄 Attempting fallback: Downloading IMDB dataset from Hugging Face...")
        
        # Fallback: Use Hugging Face datasets
        dataset = load_dataset("imdb")
        
        # Convert Train and Test sets to DataFrame
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Concatenate and reformat
        df = pd.concat([train_df, test_df], ignore_index=True)
        df.columns = ['review', 'sentiment']
        df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive'})
        
        # Save as local CSV
        df.to_csv('./data/IMDB Dataset.csv', index=False)
        print("✅ IMDB dataset successfully downloaded from Hugging Face!")
    
    print(f"📊 Total samples: {len(df)}")
    print(f"📊 Positive reviews: {len(df[df['sentiment'] == 'positive'])}")
    print(f"📊 Negative reviews: {len(df[df['sentiment'] == 'negative'])}")
    
    # Display preview
    print("\n📝 Data Preview:")
    print(df.head())
    
    return df

def download_additional_datasets():
    """Download supplemental sentiment analysis datasets for research"""
    
    print("\n🔄 Downloading additional sentiment analysis datasets...")
    
    try:
        # Twitter Sentiment Analysis (Sentiment140)
        kaggle.api.dataset_download_files(
            'kazanova/sentiment140',
            path='./data',
            unzip=True
        )
        print("✅ Twitter Sentiment140 dataset downloaded!")
        
    except Exception as e:
        print(f"⚠️ Twitter dataset download failed: {e}")
    
    try:
        # Amazon Product Reviews
        kaggle.api.dataset_download_files(
            'bittlingmayer/amazonreviews',
            path='./data',
            unzip=True
        )
        print("✅ Amazon Reviews dataset downloaded!")
        
    except Exception as e:
        print(f"⚠️ Amazon dataset download failed: {e}")

def check_gpu():
    """Verify GPU availability for accelerated training"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 Detected GPU: {gpu_name}")
        print(f"💾 Available VRAM: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️ GPU not detected; the system will default to CPU.")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Neural Sentiment Analysis Project")
    print("=" * 60)
    
    # Check GPU status
    gpu_available = check_gpu()
    
    # Download primary dataset
    df = download_imdb_dataset()
    
    # Download secondary datasets (optional)
    # download_additional_datasets()
    
    print("\n🎉 Preparation Complete!")
    print("▶️ You can now run 'python train_simple.py' to begin training.")
