# Instrument Classification and Separation

This project provides deep learning models for:

1. **Instrument Classification**: Identifying which instruments are present in a mixed audio track
2. **Instrument Separation**: Extracting individual instrument audio from a mixed track

## Project Structure

```
models/model_HL/
├── model.py              # Instrument classification model architecture
├── dataset.py            # Dataset class for loading and processing audio data
├── train.py              # Script for training the instrument classifier
├── test_model.py         # Script for testing the classification model
├── separation.py         # Source separation model architecture
├── train_separation.py   # Script for training instrument separation models
├── extract_instruments.py # Script for extracting instruments from audio files
├── download_pretrained.py # Script for downloading pretrained models (if available)
└── README.md             # This file
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. For MP3 support, make sure ffmpeg is installed on your system:

```bash
# On Ubuntu/Debian
sudo apt-get install ffmpeg

# On macOS using Homebrew
brew install ffmpeg

# On Windows using Chocolatey
choco install ffmpeg
```

## Data Structure

The system works with audio data organized as follows:

```
data/raw/
├── Track00001/
│   ├── mix.wav           # Mixed audio track
│   ├── stems/            # Individual instrument tracks
│   │   ├── S00.wav       # Instrument 1
│   │   ├── S01.wav       # Instrument 2
│   │   └── ...
│   └── metadata.yaml     # Metadata with instrument information
├── Track00002/
└── ...
```

## Usage

### 1. Training the Instrument Classifier

```bash
python train.py
```

This will train an ensemble of models to detect which instruments are present in a mixed audio track. The trained models will be saved in the `ensemble` directory.

### 2. Testing the Classifier

```bash
python test_model.py
```

This will evaluate the trained classifier on a test set and generate performance metrics.

### 3. Training the Instrument Separation Models

```bash
python train_separation.py --classifier_dir ./ --output_dir ./separation_models
```

This trains separate models for each instrument class to extract them from mixed tracks. The script will look for classifier models in the following locations:

- First in the `ensemble` directory inside the specified `classifier_dir`
- Then with standard naming patterns in the root of `classifier_dir`

### 4. Extracting Instruments from Audio

Extract instruments from a single audio file and save as WAV:

```bash
python extract_instruments.py --input path/to/mix.wav --output_dir ./extracted_stems
```

Extract instruments and save as MP3:

```bash
python extract_instruments.py --input path/to/mix.wav --output_dir ./extracted_stems --format mp3
```

Or process a whole dataset:

```bash
python extract_instruments.py --input data/raw --batch --output_dir ./extracted_stems
```

Each extracted instrument will be saved as a separate audio file, and visualizations of waveforms and spectrograms will be generated for comparison.

## Model Architecture

### Classification Model

- ResNet-based architecture
- Multi-label classification (can detect multiple instruments simultaneously)
- Uses mel spectrograms as input features

### Separation Model

- Based on Demucs architecture (simplified)
- U-Net-like encoder-decoder with skip connections
- Time-domain waveform separation
- Trained separately for each instrument class

## Performance

The classification model achieves around 85% accuracy in detecting instrument presence.

The separation quality depends on:

- Instrument type (percussion and bass are easier to separate than similar-sounding instruments)
- Amount of training data for each instrument class
- Complexity of the mix

## Advanced Configuration

Both training scripts accept various command-line arguments to customize the training process:

```bash
# View all training options for separation models
python train_separation.py --help

# View all extraction options
python extract_instruments.py --help
```

## License

[MIT License](LICENSE)
