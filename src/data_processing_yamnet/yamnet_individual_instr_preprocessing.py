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
    """Preprocess and segment audio files for YAMNet, including individual stems."""
    for track_num in range(1, 21):
        track_folder = f"Track0000{track_num}" if track_num < 10 else f"Track000{track_num}"
        raw_track_path = RAW_DATA_PATH / track_folder
        processed_track_path = PROCESSED_DATA_PATH / track_folder

        # Process mix.wav
        input_mix_path = raw_track_path / "mix.wav"
        output_mix_path = processed_track_path / "mix.wav"
        segments_folder = processed_track_path / "segments"

        processed_track_path.mkdir(parents=True, exist_ok=True)

        if input_mix_path.exists():
            preprocess_audio(input_mix_path, output_mix_path, TARGET_SR)
            split_audio(output_mix_path, segments_folder, TARGET_SR, SEGMENT_DURATION)
            print(f"Processed and segmented {input_mix_path} → {output_mix_path}")
        else:
            print(f"File not found: {input_mix_path}")

        # Process stems (S08.wav - S017.wav)
        for stem_num in range(8, 18):
            stem_name = f"S0{stem_num}"
            stem_input_path = raw_track_path / f"{stem_name}.wav"
            stem_output_folder = processed_track_path / "stems" / stem_name

            stem_output_folder.mkdir(parents=True, exist_ok=True)

            if stem_input_path.exists():
                stem_output_path = stem_output_folder / f"{stem_name}.wav"
                preprocess_audio(stem_input_path, stem_output_path, TARGET_SR)
                split_audio(stem_output_path, stem_output_folder / "segments", TARGET_SR, SEGMENT_DURATION)
                print(f"Processed and segmented {stem_input_path} → {stem_output_folder}")
            else:
                print(f"Stem file not found: {stem_input_path}")


if __name__ == "__main__":
    main()
