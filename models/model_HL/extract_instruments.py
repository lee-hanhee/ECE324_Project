import os
import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from separation import InstrumentSeparator

def plot_waveforms(mix_waveform, separated_waveforms, output_path):
    """
    Plot original mix and separated waveforms for visual comparison.
    
    Args:
        mix_waveform: Original mix waveform
        separated_waveforms: Dictionary of separated waveforms
        output_path: Path to save the plot
    """
    num_instruments = len(separated_waveforms)
    fig, axes = plt.subplots(num_instruments + 1, 1, figsize=(10, 2 * (num_instruments + 1)))
    
    # Handle case with only one instrument
    if num_instruments == 1:
        axes = [axes]
    
    # Plot mix
    axes[0].plot(mix_waveform.numpy()[0])
    axes[0].set_title('Mix')
    axes[0].set_ylim(-1, 1)
    
    # Plot each instrument
    for i, (inst, waveform) in enumerate(separated_waveforms.items(), 1):
        axes[i].plot(waveform.numpy()[0])
        axes[i].set_title(f'Extracted: {inst}')
        axes[i].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_spectrograms(mix_waveform, separated_waveforms, output_path, sample_rate=44100):
    """
    Plot spectrograms of original mix and separated waveforms.
    
    Args:
        mix_waveform: Original mix waveform
        separated_waveforms: Dictionary of separated waveforms
        output_path: Path to save the plot
        sample_rate: Audio sample rate
    """
    num_instruments = len(separated_waveforms)
    fig, axes = plt.subplots(num_instruments + 1, 1, figsize=(10, 3 * (num_instruments + 1)))
    
    # Handle case with only one instrument
    if num_instruments == 1:
        axes = [axes]
    
    # Calculate spectrogram parameters
    n_fft = 2048
    hop_length = 512
    
    # Plot mix spectrogram
    mix_spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2
    )(mix_waveform)
    
    axes[0].imshow(
        torch.log10(mix_spec[0] + 1e-9),
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    axes[0].set_title('Mix Spectrogram')
    axes[0].set_ylabel('Frequency')
    
    # Plot each instrument spectrogram
    for i, (inst, waveform) in enumerate(separated_waveforms.items(), 1):
        spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2
        )(waveform)
        
        axes[i].imshow(
            torch.log10(spec[0] + 1e-9),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        axes[i].set_title(f'{inst} Spectrogram')
        axes[i].set_ylabel('Frequency')
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_as_mp3(waveform, filepath, sample_rate):
    """
    Save a waveform as an MP3 file.
    
    Args:
        waveform: Audio waveform tensor
        filepath: Path to save the MP3 file
        sample_rate: Audio sample rate
    """
    try:
        import soundfile as sf
        import subprocess
        
        # First save as temporary WAV file
        temp_wav = filepath.replace('.mp3', '_temp.wav')
        torchaudio.save(temp_wav, waveform, sample_rate)
        
        # Use ffmpeg to convert WAV to MP3
        subprocess.call(['ffmpeg', '-i', temp_wav, '-codec:a', 'libmp3lame', '-qscale:a', '2', filepath, '-y', '-loglevel', 'quiet'])
        
        # Remove temporary WAV file
        os.remove(temp_wav)
        
        return True
    except Exception as e:
        print(f"Error converting to MP3: {e}")
        return False

def extract_instruments(
    input_path,
    output_dir,
    classifier_dir,
    separation_dir,
    sample_rate=44100,
    device=None,
    output_format='wav'
):
    """
    Extract instruments from an audio file.
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory to save extracted instruments
        classifier_dir: Directory containing classifier models
        separation_dir: Directory containing separation models
        sample_rate: Audio sample rate
        device: Device to run inference on
        output_format: Output audio format ('wav' or 'mp3')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find classifier models
    classifier_paths = []
    
    # Try ensemble directory first (most likely location)
    ensemble_dir = os.path.join(classifier_dir, 'ensemble')
    if os.path.exists(ensemble_dir) and os.path.isdir(ensemble_dir):
        # Look for model files in the ensemble directory
        model_files = [os.path.join(ensemble_dir, f) for f in os.listdir(ensemble_dir) 
                      if f.endswith('.pth')]
        if model_files:
            classifier_paths = model_files
            print(f"Found models in ensemble directory: {classifier_paths}")
    
    # If no models found in ensemble, try the standard locations
    if not classifier_paths:
        # Try ensemble models first
        ensemble_models = [os.path.join(classifier_dir, f"ensemble_model_{i+1}.pth") for i in range(5)]
        existing_ensemble_models = [path for path in ensemble_models if os.path.exists(path)]
        
        if existing_ensemble_models:
            classifier_paths = existing_ensemble_models
        else:
            # Try fold models
            fold_models = [os.path.join(classifier_dir, f"best_model_fold_{i+1}.pth") for i in range(5)]
            existing_fold_models = [path for path in fold_models if os.path.exists(path)]
            if existing_fold_models:
                classifier_paths = existing_fold_models
    
    if not classifier_paths:
        raise ValueError("No trained classifier models found. Please train the classifier first.")
    
    print(f"Found {len(classifier_paths)} classifier model(s)")
    
    # Determine instrument classes from existing separation models
    instrument_classes = []
    for filename in os.listdir(separation_dir):
        if filename.endswith('_separator.pth'):
            instrument = filename.replace('_separator.pth', '').replace('_', ' ')
            instrument_classes.append(instrument)
    
    print(f"Found separation models for {len(instrument_classes)} instruments: {instrument_classes}")
    
    # Create instrument separator
    separator = InstrumentSeparator(
        classifier_paths=classifier_paths,
        instrument_classes=instrument_classes,
        sample_rate=sample_rate,
        device=device
    )
    
    # Load separation models
    separator.load_separation_models(separation_dir)
    
    # Load the audio file
    waveform, sr = torchaudio.load(input_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Classify instruments
    instrument_predictions = separator.classify_instruments(waveform)
    predicted_instruments = [instrument_classes[i] for i in range(len(instrument_classes)) 
                            if instrument_predictions[i] > 0.5]
    
    print(f"Detected instruments: {', '.join(predicted_instruments)}")
    
    # Separate instruments
    separated_waveforms = separator.separate_instruments(waveform, instrument_predictions)
    
    # Save separated waveforms
    output_paths = {}
    for instrument, separated in separated_waveforms.items():
        file_ext = f".{output_format.lower()}"
        output_path = os.path.join(output_dir, f"{instrument.replace(' ', '_')}{file_ext}")
        
        if output_format.lower() == 'mp3':
            if save_as_mp3(separated, output_path, sample_rate):
                output_paths[instrument] = output_path
                print(f"Saved separated {instrument} to {output_path}")
        else:
            # Default to WAV
            torchaudio.save(output_path, separated, sample_rate)
            output_paths[instrument] = output_path
            print(f"Saved separated {instrument} to {output_path}")
    
    # Plot waveforms and spectrograms for visual comparison
    if separated_waveforms:
        # Plot waveforms
        plot_waveforms(
            waveform,
            separated_waveforms,
            os.path.join(output_dir, 'waveform_comparison.png')
        )
        
        # Plot spectrograms
        plot_spectrograms(
            waveform,
            separated_waveforms,
            os.path.join(output_dir, 'spectrogram_comparison.png'),
            sample_rate
        )
        
        print(f"Visualizations saved to {output_dir}/waveform_comparison.png and {output_dir}/spectrogram_comparison.png")
    
    print(f"Extracted {len(output_paths)} instruments from {input_path}")
    print(f"Results saved to {output_dir}")
    
    return output_paths

def extract_instruments_batch(
    data_dir,
    output_dir,
    classifier_dir,
    separation_dir,
    sample_rate=44100,
    device=None,
    limit=None,
    output_format='wav'
):
    """
    Extract instruments from multiple audio files in a directory.
    
    Args:
        data_dir: Directory containing audio files
        output_dir: Directory to save extracted instruments
        classifier_dir: Directory containing classifier models
        separation_dir: Directory containing separation models
        sample_rate: Audio sample rate
        device: Device to run inference on
        limit: Maximum number of files to process
        output_format: Output audio format ('wav' or 'mp3')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all track directories
    track_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if limit is not None:
        track_dirs = track_dirs[:limit]
    
    print(f"Found {len(track_dirs)} tracks to process")
    
    # Process each track
    for i, track_dir in enumerate(track_dirs):
        print(f"\nProcessing track {i+1}/{len(track_dirs)}: {track_dir}")
        
        # Get path to mix.wav
        mix_path = os.path.join(data_dir, track_dir, 'mix.wav')
        if not os.path.exists(mix_path):
            print(f"Warning: No mix.wav found for {track_dir}, skipping")
            continue
        
        # Create output directory for this track
        track_output_dir = os.path.join(output_dir, track_dir, 'extracted_stems')
        os.makedirs(track_output_dir, exist_ok=True)
        
        # Extract instruments
        try:
            extract_instruments(
                input_path=mix_path,
                output_dir=track_output_dir,
                classifier_dir=classifier_dir,
                separation_dir=separation_dir,
                sample_rate=sample_rate,
                device=device,
                output_format=output_format
            )
        except Exception as e:
            print(f"Error processing {track_dir}: {e}")
    
    print(f"\nProcessed {len(track_dirs)} tracks")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract instruments from audio files')
    parser.add_argument('--input', type=str, help='Path to input audio file or directory')
    parser.add_argument('--output_dir', type=str, default='./extracted_stems', 
                        help='Directory to save extracted instruments')
    parser.add_argument('--classifier_dir', type=str, default='./', 
                        help='Directory containing classifier models')
    parser.add_argument('--separation_dir', type=str, default='./separation_models', 
                        help='Directory containing separation models')
    parser.add_argument('--sample_rate', type=int, default=44100, 
                        help='Audio sample rate')
    parser.add_argument('--batch', action='store_true', 
                        help='Process all tracks in the input directory')
    parser.add_argument('--limit', type=int, 
                        help='Maximum number of files to process in batch mode')
    parser.add_argument('--format', type=str, choices=['wav', 'mp3'], default='wav',
                        help='Output audio format (wav or mp3)')
    args = parser.parse_args()
    
    # Get project root if input not specified
    if args.input is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../..'))
        args.input = os.path.join(project_root, 'data', 'raw')
        args.batch = True
    
    if args.batch:
        # Process all tracks in the directory
        extract_instruments_batch(
            data_dir=args.input,
            output_dir=args.output_dir,
            classifier_dir=args.classifier_dir,
            separation_dir=args.separation_dir,
            sample_rate=args.sample_rate,
            limit=args.limit,
            output_format=args.format
        )
    else:
        # Process a single audio file
        extract_instruments(
            input_path=args.input,
            output_dir=args.output_dir,
            classifier_dir=args.classifier_dir,
            separation_dir=args.separation_dir,
            sample_rate=args.sample_rate,
            output_format=args.format
        )

if __name__ == '__main__':
    main() 