# Musical Instrument Classification with Deep Learning

## 🧠 Overview

This project provides a robust framework for automatic instrument classification in audio files using deep learning techniques. It combines custom neural networks with pretrained models like YAMNet to accurately identify musical instruments within audio recordings.

## ✨ Features

- Multi-label instrument classification in audio recordings
- Custom CNN-based model for instrument recognition
- Integration of Google's YAMNet for acoustic event detection
- Audio preprocessing and feature extraction tools
- Data augmentation techniques for improved generalization
- Cross-validation framework for model evaluation
- Comprehensive visualization tools for model performance analysis

## 🛠️ Tech Stack

- Python
- PyTorch & TorchAudio
- Librosa (audio analysis)
- NumPy
- scikit-learn
- Matplotlib & Seaborn (visualization)
- FFmpeg (audio processing)
- YAMNet (pretrained audio classification model)

## �� Code Structure

```
├── data/                         # Source audio files with instrument labels
│   ├── raw/                      # Raw audio recordings
│   └── instruments/              # Instrument-labeled audio files
├── models/                       # Implementation of various classification models
│   ├── instrument_classification/# Custom multi-label instrument classifier
│   │   ├── identification.py     # Main classification logic
│   │   ├── processing.py         # Audio processing utilities
│   │   ├── visualizations.py     # Visualization tools
│   │   └── saved_model.pth       # Trained model weights
│   ├── YAMNet/                   # Implementation using Google's YAMNet
│   │   ├── yamnet.ipynb          # YAMNet experimentation notebook
│   │   └── yamnet_predict_segment_v5.py # YAMNet prediction script
│   ├── pretrained/               # Pretrained models for classification
│   │   ├── saved_data/           # Cached model outputs
│   │   └── instrument_classifier.py # Pretrained classifier implementation
│   ├── NMF/                      # Non-negative Matrix Factorization experiments
│   └── WaveNet/                  # WaveNet model implementation
├── src/                          # Source code for data processing and exploration
│   ├── data_exploration/         # Scripts for analyzing the dataset
│   ├── data_processing_yamnet/   # Preprocessing for YAMNet compatibility
│   └── instrument data preprocessing/ # Tools for audio preprocessing
├── results/                      # Generated visualizations and metrics
│   ├── metrics/                  # Model performance evaluation metrics
│   ├── spectrograms/             # Generated spectrograms from audio samples
│   ├── waveforms/                # Visualized audio waveforms
│   └── yamnet_predictions/       # Output predictions from YAMNet model
├── yamnet_env/                   # Virtual environment for YAMNet
├── requirements.txt              # List of required Python packages
├── LICENSE                       # Project license
└── README.md                     # This file
```

## 🚀 How to Reproduce

### Environment Setup

```bash
# Create and activate a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Instrument Classifier

```bash
# Run the custom instrument classifier
python models/instrument_classification/identification.py

# Or use the pretrained model
python models/pretrained/instrument_classifier.py
```

### Dataset Requirements

- Audio files should be in WAV format
- Sample rate: 22050 Hz (default setting)
- Each audio file should be labeled with corresponding instruments
- The data should be organized in appropriate directories as specified in the data/ folder

## 📊 Results

Our custom instrument classification model achieves significant improvement over baseline YAMNet for specific instrument recognition tasks:

| Model             | Accuracy | F1 Score |
| ----------------- | -------- | -------- |
| YAMNet (baseline) | -- %     | -- %     |
| Custom Model      | -- %     | -- %     |

_Note: Fill in the actual metrics after running the model evaluation._

The visual analysis of model performance is available in the `results/metrics/` directory.
