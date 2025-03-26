import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pathlib import Path


class YAMNetClassifier:
    """Classify sounds using YAMNet pre-trained model."""

    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = self.model.class_map_path().numpy()
        self.class_names = self._class_names_from_csv(class_map_path)

    @staticmethod
    def _class_names_from_csv(class_map_csv_text):
        """Extract class names from the CSV file."""
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    @staticmethod
    def _ensure_sample_rate(sample_rate, waveform, desired_sample_rate=16000):
        """Resample waveform to desired_sample_rate if needed."""
        if sample_rate != desired_sample_rate:
            raise ValueError(f'Sample rate must be {desired_sample_rate} Hz, got {sample_rate} Hz instead.')
        return waveform

    def classify_sound(self, wav_file_path, result_fig_path):
        """Classify the main sound from the provided WAV file and save figure."""
        sample_rate, wav_data = wavfile.read(wav_file_path)
        wav_data = self._ensure_sample_rate(sample_rate, wav_data)
        waveform = wav_data / tf.int16.max

        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()

        inferred_class = self.class_names[scores_np.mean(axis=0).argmax()]

        print(f'File: {os.path.basename(wav_file_path)} classified as: {inferred_class}')

        self._visualize_results(waveform, spectrogram_np, scores_np, result_fig_path)

    def _visualize_results(self, waveform, spectrogram_np, scores_np, result_fig_path, top_n=10):
        """Visualize waveform, spectrogram, and top predicted classes."""
        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(waveform)
        plt.title('Waveform')
        plt.xlim([0, len(waveform)])

        plt.subplot(3, 1, 2)
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
        plt.title('Spectrogram')

        mean_scores = np.mean(scores_np, axis=0)
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]

        plt.subplot(3, 1, 3)
        plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
        plt.yticks(range(top_n), [self.class_names[i] for i in top_class_indices])
        plt.title('Top Class Predictions')

        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding - 0.5, scores_np.shape[0] + patch_padding - 0.5])

        plt.tight_layout()
        plt.savefig(result_fig_path)
        plt.close()

if __name__ == "__main__":
    classifier = YAMNetClassifier()

    base_dir = Path('data/processed/yamnet/')
    result_base_dir = Path('results/yamnet_predictions/')

    for track_num in range(1, 21):
        track_folder = f"Track0000{track_num}" if track_num < 10 else f"Track000{track_num}"
        segments_folder = base_dir / track_folder / 'segments'
        result_folder = result_base_dir / track_folder

        result_folder.mkdir(parents=True, exist_ok=True)

        if segments_folder.exists():
            for segment_file in sorted(segments_folder.glob('segment_*.wav')):
                segment_number = segment_file.stem.split('_')[1]
                result_fig_path = result_folder / f"segment_{segment_number}.png"
                classifier.classify_sound(segment_file, result_fig_path)
        else:
            print(f'Segments not found: {segments_folder}')