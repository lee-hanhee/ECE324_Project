import numpy as np
import csv

import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
import librosa
import librosa.display
import seaborn as sns


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

def plot_waveform(y, sr, track_name):
  plt.figure(figsize=(10, 4))
  librosa.display.waveshow(y, sr=sr)
  plt.title('Waveform of the Audio')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  image_path = "results/waveforms/" + track_name + "-waveforms.png"
  plt.savefig(image_path)

def calculate_rms(audio):
    return np.sqrt(np.mean(np.square(audio)))

def plot_boxplot_histogram(rms_values):
    plt.figure(figsize=(10, 6))
    plt.hist(rms_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of RMS (Volume) Levels')
    plt.xlabel('RMS (Volume Level)')
    plt.ylabel('Frequency')
    image_path = "results/volume_boxplot.png"
    plt.savefig(image_path)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=rms_values, color='skyblue')
    plt.title('Boxplot of RMS (Volume) Levels')
    plt.ylabel('RMS (Volume Level)')
    image_path = "results/volume_histogram.png"
    plt.savefig(image_path)

if __name__ == "__main__":
  base_path = "data/raw"
  rms = []
  for i in range(1, 21):  # Tracks are numbered from 1 to 20
    track_name = f"Track{i:05d}"
    wav_file = Path(base_path) / track_name  / "mix.wav"
    sample_rate, wav_data = wavfile.read(wav_file, 'rb')
    y, sr = librosa.load(wav_file)
    # plot_spectrogram(wav_data, sample_rate, track_name)
    # plot_waveform(y, sr, track_name)
    rms.append(calculate_rms(y))
    print("Done " + track_name)
  plot_boxplot_histogram(rms)

