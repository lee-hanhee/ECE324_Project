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
- TensorFlow & TensorFlow Hub (YAMNet model)
- UMAP (dimensionality reduction for data exploration)

## Code Structure

```
├── data/                           # Source audio files with instrument labels
│   ├── raw/                        # Raw audio recordings from BabySlakH dataset
│   └── instruments/                # Instrument audio files for each instrument
├── experiments/                    # Experimental notebooks and scripts
│   ├── MultilabelClassification.ipynb  # Multi-label classification experiments
│   ├── wavenet.ipynb                   # WaveNet experimentation
│   └── yamnet_example.ipynb            # YAMNet examples and implementation
├── models/                         # Implementation of various classification models
│   ├── instrument_classification/  # Custom multi-label instrument classifier
│   │   ├── identification.py       # Main instrument identification implementation
│   │   ├── processing.py           # Audio processing for identification
│   │   ├── visualizations.py       # Visualization tools for model results
│   │   ├── baseline.py             # Baseline classifier implementation
│   │   ├── model_v1.pth            # Trained model weights (version 1)
│   │   ├── model_v2.pth            # Trained model weights (version 2)
│   │   ├── saved_model.pth         # Latest trained model weights
│   │   └── results/                # Classification results and metrics
│   ├── NMF/                        # Non-negative Matrix Factorization experiments
│   ├── SpectrogramUNet/            # Spectrogram U-Net model implementation
│   ├── YAMNet/                     # YAMNet-based models and utilities
│   │   ├── checkpoints/            # Saved model checkpoints
│   │   ├── pretrained_no_fine_tuning/ # Pretrained model implementations
│   │   ├── results/                # Results and visualizations
│   │   ├── utils/                  # Utility functions for YAMNet
│   │   └── instrument_classifier_v2.py # Enhanced classifier with detailed metrics
│   └── comparison.py               # Script for comparing model performance
├── results/                        # Generated visualizations and metrics
│   ├── metrics/                    # Model performance evaluation metrics
│   │   ├── track_instruments.csv   # Instrument labels for each track
│   │   ├── inst_counts.txt         # Counts of instruments in dataset
│   │   └── data_summary.csv        # Summary statistics of the dataset
│   ├── plots/                      # Generated plots and visualizations
│   │   ├── umap/                   # UMAP projections of audio features
│   │   └── inst_dist/              # Instrument distribution visualizations
│   ├── spectrograms/               # Generated spectrograms for analysis
│   ├── waveforms/                  # Audio waveform visualizations
│   ├── yamnet_predictions/         # Output predictions from YAMNet model
│   ├── Audio Frequency Ranges.png  # Audio frequency range analysis
│   ├── AudioTrackDurations.png     # Track duration statistics
│   ├── InstruNETYamNETf1.png       # F1 score comparison visualization
│   └── volume_boxplot.png          # Box plot of volume statistics
│   ├── volume_histogram.png        # Volume distribution analysis
├── src/                            # Source code for data processing and exploration
│   ├── data_exploration/           # Scripts for analyzing the dataset
│   │   ├── audio_exploration.py    # Audio data analysis and visualization
│   │   ├── figures.py              # Visualization utilities and plot functions
│   │   ├── instrument_counts.py    # Analysis of instrument distribution
│   │   ├── instrumentation.py      # Tools for instrument data analysis
│   │   ├── spectrogam.py           # Spectrogram generation and analysis
│   │   ├── umap_exploration_basic.py  # Basic UMAP dimensionality reduction
│   │   └── umap_exploration_mfcc.py   # UMAP visualization with MFCCs
│   ├── data_processing_yamnet/     # Preprocessing for YAMNet compatibility
│   │   ├── yamnet_preprocessing.py            # YAMNet audio preprocessing
│   │   └── yamnet_individual_instr_preprocessing.py  # Instrument-specific processing
│   └── instrument data preprocessing/ # Tools for audio preprocessing
│       ├── instrument_extraction.py  # Extraction of individual instruments
│       └── combined_data.py          # Combining and processing instrument data
├── requirements.txt                # List of required Python packages
├── LICENSE                         # Project license
└── README.md                       # This file
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

### Running the YAMNet Classifier

```bash
# Using the pretrained YAMNet segment predictor
python models/yamnet/pretrained/segment_predictor.py

# Or training an instrument classifier with YAMNet embeddings
python models/yamnet/core/instrument_classifier.py

# Or using the enhanced classifier with detailed metrics
python models/yamnet/core/instrument_classifier_v2.py
```

### Processing Audio for YAMNet

```bash
# Process audio files for YAMNet
python src/data_processing_yamnet/yamnet_preprocessing.py

# Process individual instrument stems for YAMNet
python src/data_processing_yamnet/yamnet_individual_instr_preprocessing.py
```

### Data Exploration

```bash
# Generate UMAP visualizations
python src/data_exploration/umap_exploration_mfcc.py

# Explore instrument distributions
python src/data_exploration/instrument_counts.py

# Generate spectrograms
python src/data_exploration/spectrogam.py
```

### Dataset Requirements

- Audio files should be in WAV format
- Sample rate: 16kHz for YAMNet, 22050 Hz for other models
- Each audio file should be labeled with corresponding instruments
- The data should be organized in appropriate directories as specified in the data/ folder

## 📊 Results

Our instrument classification models achieve significant accuracy for instrument recognition tasks:

| Model                  | Accuracy |
| ---------------------- | -------- |
| YAMNet + LogReg        | ~73%     |
| Multi-label Classifier | ~82%     |

_Note: The exact metrics will vary based on your specific dataset and model configurations._

The visual analysis of model performance is available in the `models/yamnet/results/metrics/` directory.

## 📝 License

This project is licensed under the terms of the license included in the repository.
