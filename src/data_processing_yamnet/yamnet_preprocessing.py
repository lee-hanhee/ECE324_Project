import os
from pathlib import Path

import librosa
import soundfile as sf

# Assumes current directory is the root of the project
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed/yamnet")
TARGET_SR = 16000  # Sample rate in Hz
SEGMENT_DURATION = 20  # Seconds


def preprocess_audio(input_path: Path, output_path: Path, target_sr: int):
    """
    Load audio, resample to mono at target sample rate, and save as WAV.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio
        target_sr: Target sample rate in Hz
    """
    audio, _ = librosa.load(input_path, sr=target_sr, mono=True)
    sf.write(output_path, audio, samplerate=target_sr)


def split_audio(audio_path: Path, segment_folder: Path, target_sr: int, segment_duration: int):
    """
    Split audio file into segments of equal duration.
    
    Args:
        audio_path: Path to audio file to split
        segment_folder: Folder to save segments
        target_sr: Sample rate of the audio
        segment_duration: Duration of each segment in seconds
    """
    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    segment_samples = segment_duration * target_sr

    segment_folder.mkdir(parents=True, exist_ok=True)

    for idx, start in enumerate(range(0, len(audio), segment_samples)):
        segment_audio = audio[start:start + segment_samples]
        segment_filename = segment_folder / f"segment_{idx + 1}.wav"
        sf.write(segment_filename, segment_audio, samplerate=target_sr)


def main():
    """Preprocess all audio files for YAMNet and split into segments."""
    for track_num in range(1, 21):
        track_folder = f"Track0000{track_num}" if track_num < 10 else f"Track000{track_num}"

        input_filename = RAW_DATA_PATH / track_folder / "mix.wav"
        output_folder = PROCESSED_DATA_PATH / track_folder
        output_filename = output_folder / "mix.wav"
        segments_folder = output_folder / "segments"

        output_folder.mkdir(parents=True, exist_ok=True)

        if input_filename.exists():
            preprocess_audio(input_filename, output_filename, TARGET_SR)
            split_audio(output_filename, segments_folder, TARGET_SR, SEGMENT_DURATION)
            print(f"Processed and segmented {input_filename} → {output_filename}")
        else:
            print(f"File not found: {input_filename}")


if __name__ == "__main__":
    main()