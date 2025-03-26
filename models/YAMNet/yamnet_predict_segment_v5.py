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
        """Ensure the waveform is at the desired sample rate."""
        if sample_rate != desired_sample_rate:
            raise ValueError(f'Sample rate must be {desired_sample_rate} Hz, got {sample_rate} Hz instead.')
        return waveform

    def classify_sound(self, wav_file_path, result_fig_path):
        """Classify the main sound from the provided WAV file and save the figure."""
        sample_rate, wav_data = wavfile.read(wav_file_path)
        wav_data = self._ensure_sample_rate(sample_rate, wav_data)
        waveform = wav_data / tf.int16.max

        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()

        inferred_class = self.class_names[scores_np.mean(axis=0).argmax()]
        print(f'File: {os.path.basename(wav_file_path)} classified as: {inferred_class}')

        self._visualize_results(waveform, spectrogram_np, scores_np, result_fig_path, sample_rate)

    def _visualize_results(self, waveform, spectrogram_np, scores_np, result_fig_path, sample_rate, top_n=10):
        """Visualize waveform, spectrogram, and top predicted classes with labels."""
        plt.figure(figsize=(12, 9))

        # Time axis for waveform
        time_axis = np.arange(len(waveform)) / sample_rate

        plt.subplot(3, 1, 1)
        plt.plot(time_axis, waveform)
        plt.title('Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude (normalized)')
        plt.xlim([0, time_axis[-1]])

        # Spectrogram visualization
        plt.subplot(3, 1, 2)
        spectrogram_time_axis = np.linspace(0, time_axis[-1], spectrogram_np.shape[0])
        spectrogram_freq_axis = np.linspace(0, sample_rate / 2, spectrogram_np.shape[1])  # Nyquist frequency
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower',
                   extent=[spectrogram_time_axis[0], spectrogram_time_axis[-1], spectrogram_freq_axis[0], spectrogram_freq_axis[-1]])
        plt.title('Spectrogram')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')

        # Top class predictions visualization
        mean_scores = np.mean(scores_np, axis=0)
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]  # Get top N classes

        plt.subplot(3, 1, 3)
        frames = np.arange(scores_np.shape[0])

        # Limit to the first 10 frames
        frame_limit = min(10, scores_np.shape[0])

        # Flip class order so that the most confident predictions appear at the top
        plt.imshow(scores_np[:frame_limit, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r',
           extent=[0, frame_limit, 0, top_n])
        plt.gca().invert_yaxis()  # Correct the y-axis so top labels match top predictions
        plt.yticks(range(top_n), [self.class_names[i] for i in top_class_indices[::-1]])  # Flip label order
        plt.title('Top Class Predictions')
        plt.xlabel('Frames')
        plt.ylabel('Predicted Classes')

        plt.tight_layout()
        plt.savefig(result_fig_path)
        plt.close()


if __name__ == "__main__":
    classifier = YAMNetClassifier()

    base_dir = Path('raw/')
    result_base_dir = Path('results/yamnet_predictions/')

    for track_num in range(1, 21):
        track_folder = f"Track0000{track_num}" if track_num < 10 else f"Track000{track_num}"
        track_path = base_dir / track_folder
        mix_file = track_path / 'mix.wav'
        result_folder = result_base_dir / track_folder

        result_folder.mkdir(parents=True, exist_ok=True)

        if mix_file.exists():
            result_fig_path = result_folder / "mix_prediction.png"
            classifier.classify_sound(mix_file, result_fig_path)
        else:
            print(f'Mix file not found: {mix_file}')
