import os
from pathlib import Path
import librosa
import soundfile as sf

# Assumes current directory is the root of the project
RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed/yamnet")
TARGET_SR = 16000
SEGMENT_DURATION = 20  # seconds


def preprocess_audio(input_path: Path, output_path: Path, target_sr: int):
    """Load audio, resample to mono at target sample rate, and save as WAV."""
    audio, _ = librosa.load(input_path, sr=target_sr, mono=True)
    sf.write(output_path, audio, samplerate=target_sr)


def split_audio(audio_path: Path, segment_folder: Path, target_sr: int, segment_duration: int):
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
        output_mix_path = processed_track_path = PROCESSED_DATA_PATH / track_folder / "mix.wav"
        segments_folder = output_folder / "segments"

        output_folder.mkdir(parents=True, exist_ok=True)

        if input_filename.exists():
            preprocess_audio(input_filename, output_filename, TARGET_SR)
            split_audio(output_filename, segments_folder, TARGET_SR, SEGMENT_DURATION)
            print(f"Processed and segmented {input_filename} → {output_filename}")
        else:
            print(f"File not found: {input_filename}")

        # Process stems (S08.wav - S017.wav)
        for stem_num in range(8, 18):
            stem_name = f"S0{stem_num}"
            stem_input_path = RAW_DATA_PATH / track_folder / f"{stem_name}.wav"
            stem_output_folder = PROCESSED_DATA_PATH / track_folder / "stems" / stem_name

            if stem_input_path.exists():
                preprocess_audio(stem_input_path, stem_output_folder / f"{stem_name}.wav", TARGET_SR)
                split_audio(stem_output_folder / f"{stem_name}.wav", stem_output_folder / "segments", TARGET_SR, SEGMENT_DURATION)
                print(f"Processed and segmented {stem_input_path} → {stem_output_folder}")
            else:
                print(f"Stem file not found: {stem_input_path}")


if __name__ == "__main__":
    main()
