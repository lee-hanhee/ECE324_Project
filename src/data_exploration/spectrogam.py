import numpy as np
import csv

import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path


def save_info(wav_data, sample_rate):
  duration = len(wav_data)/sample_rate
  print(f'Sample rate: {sample_rate} Hz')
  print(f'Total duration: {duration:.2f}s')
  print(f'Size of the input: {len(wav_data)}')
  return wav_data, sample_rate

def plot_spectrogram(wav_data, sample_rate, track_name):
  plt.figure(figsize=(10, 6))
  plt.specgram(wav_data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
  plt.title('Spectrogram of mix.wav')
  plt.xlabel('Time (s)')
  plt.ylabel('Frequency (Hz)')
  plt.colorbar(label='Intensity (dB)')
  plt.tight_layout()
  image_path = "results/spectrograms/" + track_name + "-spectrogram.png"
  plt.savefig(image_path)

if __name__ == "__main__":
  base_path = "data/raw"
  for i in range(1, 21):  # Tracks are numbered from 1 to 20
    track_name = f"Track{i:05d}"
    wav_file = Path(base_path) / track_name  / "mix.wav"
    sample_rate, wav_data = wavfile.read(wav_file, 'rb')
    plot_spectrogram(wav_data, sample_rate, track_name)

