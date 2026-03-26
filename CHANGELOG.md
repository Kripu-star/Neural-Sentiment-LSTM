# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-26

### Added
- **Bidirectional LSTM Architecture:** Implemented a many-to-one recurrent neural network using PyTorch for advanced context retention.
- **Streamlit Dashboard:** Developed an interactive web interface for real-time sentiment inference and linguistic analysis.
- **Automated Preprocessing:** Added a robust pipeline for alphanumeric tokenization and vocabulary serialization (`vocabulary.pkl`).
- **Performance Visualization:** Integrated Plotly for dynamic sentiment probability charts and Matplotlib for training loss/accuracy curves.
- **Model Weights:** Included pre-trained `.pth` weights for the IMDB 50K dataset achieving 87.2% test accuracy.

### Changed
- **Local Training Optimization:** Modified data loaders and subsampling logic to allow efficient training on consumer-grade hardware (optimized for 8GB VRAM).

### Fixed
- **Dependency Resolution:** Resolved Kaggle API authentication issues by implementing a manual CSV data-loading pathway.
- **Inference Logic:** Fixed tensor device mismatches to allow seamless switching between CPU and CUDA (GPU) during web app deployment.
- **Tokenization:** Corrected padding and truncation logic to handle sequences longer than 256 tokens without crashing.

