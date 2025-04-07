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
│   ├── raw/                        # Raw audio recordings
│   └── processed/                  # Processed audio files for models
│       └── yamnet/                 # Audio processed for YAMNet
├── models/                         # Implementation of various classification models
│   ├── yamnet/                     # YAMNet-based models and utilities
│   │   ├── core/                   # Core classifier implementations
│   │   │   ├── instrument_classifier.py    # Base classifier using YAMNet embeddings
│   │   │   └── instrument_classifier_v2.py # Enhanced classifier with detailed metrics
│   │   ├── pretrained/             # Pretrained model implementations
│   │   │   ├── segment_predictor.py        # Audio segment classification with YAMNet
│   │   │   └── yamnet_example.ipynb        # YAMNet example notebook
│   │   ├── utils/                  # Utility functions
│   │   │   └── metrics.py                  # Tools for calculating weighted metrics
│   │   ├── data/                   # Processed data storage
│   │   ├── results/                # Results and visualizations
│   │   │   ├── metrics/            # Performance metrics
│   │   │   ├── confusion_matrices/ # Confusion matrix visualizations
│   │   │   └── predictions/        # YAMNet prediction results
│   │   └── README.md               # Documentation for YAMNet models
│   ├── WaveNet/                    # WaveNet model implementation
│   │   └── SpectrogramUNet.ipynb   # U-Net model for spectrograms
│   ├── NMF/                        # Non-negative Matrix Factorization experiments
│   │   └── nmf_trial.ipynb         # NMF experimentation notebook
│   └── instrument_classification/  # Custom multi-label instrument classifier
├── src/                            # Source code for data processing and exploration
│   ├── data_exploration/           # Scripts for analyzing the dataset
│   │   ├── umap_exploration_mfcc.py    # UMAP visualization with MFCCs
│   │   ├── umap_exploration_basic.py   # Basic UMAP visualization
│   │   ├── spectrogam.py              # Spectrogram generation and analysis
│   │   ├── instrument_counts.py       # Analysis of instrument distribution
│   │   ├── instrumentation.py         # Tools for instrument data analysis
│   │   ├── figures.py                 # Visualization utilities
│   │   └── audio_exploration.py       # Audio data exploration
│   ├── data_processing_yamnet/     # Preprocessing for YAMNet compatibility
│   │   ├── yamnet_preprocessing.py            # YAMNet audio preprocessing
│   │   └── yamnet_individual_instr_preprocessing.py  # Instrument-specific processing
│   └── instrument data preprocessing/ # Tools for audio preprocessing
├── experiments/                    # Experimental notebooks and scripts
│   ├── MultilabelClassification.ipynb  # Multi-label classification experiments
│   └── wavenet.ipynb                   # WaveNet experimentation
├── results/                        # Generated visualizations and metrics
│   ├── plots/                      # Generated plots and visualizations
│   │   └── umap/                   # UMAP projections of audio features
│   ├── metrics/                    # Model performance evaluation metrics
│   └── yamnet_predictions/         # Output predictions from YAMNet model
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
| YAMNet + MLP           | ~38%     | 
| Multi-label Classifier | ~78%     | 

_Note: The exact metrics will vary based on your specific dataset and model configurations._

The visual analysis of model performance is available in the `models/yamnet/results/metrics/` directory.

## 📝 License

This project is licensed under the terms of the license included in the repository.

## 📞 Contact

For questions or feedback about this project, please open an issue in the repository.
