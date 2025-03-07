import os
from pathlib import Path
import librosa
import soundfile as sf

# Assumes current directory is the root of the project
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed/yamnet")
TARGET_SR = 16000


def preprocess_audio(input_path: Path, output_path: Path, target_sr: int):
    """Load audio, resample to mono at target sample rate, and save as WAV."""
    audio, _ = librosa.load(input_path, sr=target_sr, mono=True)
    sf.write(output_path, audio, samplerate=target_sr)


def main():
    """Preprocess all audio files for YAMNet."""
    for track_num in range(1, 21):
        track_folder = f"Track0000{track_num}" if track_num < 10 else f"Track000{track_num}"

        input_filename = RAW_DATA_PATH / track_folder / "mix.wav"
        output_folder = PROCESSED_DATA_PATH / track_folder
        output_filename = output_folder / "mix.wav"

        output_folder.mkdir(parents=True, exist_ok=True)

        if input_filename.exists():
            preprocess_audio(input_filename, output_filename, TARGET_SR)
            print(f"Processed {input_filename} → {output_filename}")
        else:
            print(f"File not found: {input_filename}")


if __name__ == "__main__":
    main()