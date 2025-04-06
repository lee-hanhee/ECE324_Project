import librosa
import librosa.display
import soundfile as sf
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from figures import save_to_summary
from jaxtyping import Float, Array


def get_frequency_range(y: Float[Array, "samples"], sr: int) -> tuple[float, float]:
    """
    Calculate the minimum and maximum frequency of an audio signal.

    Parameters:
        y (ndarray): Audio time series data.
        sr (int): Sample rate of the audio signal.

    Returns:
        min_freq (float): Minimum frequency in Hz.
        max_freq (float): Maximum frequency in Hz.
    """
    D = np.abs(librosa.stft(y))  # Compute Short-Time Fourier Transform (STFT)
    frequencies = librosa.fft_frequencies(sr=sr)

    # Compute the magnitude spectrum
    avg_spectrum = np.mean(D, axis=1)  # Average across time frames
    threshold = (
        np.max(avg_spectrum) * 0.01
    )  # Consider significant frequencies above 1% of max energy

    # Find frequency range
    valid_freqs = frequencies[avg_spectrum > threshold]
    if len(valid_freqs) > 0:
        min_freq = np.min(valid_freqs)
        max_freq = np.max(valid_freqs)
    else:
        min_freq, max_freq = 0, 0  # If no valid frequencies are found

    return min_freq, max_freq


def get_audio_duration(file_path: Path) -> float:
    """
    Get duration of an audio file in seconds.
    Parameters:
        file_path (Path): Path to the audio file.
    Returns:
        float: Duration in seconds.
    """
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate


# Function to plot durations
def plot_durations(df_durations):
    """
    Plot the durations of audio tracks.
    Parameters:
        df_durations (DataFrame): DataFrame containing track names and their durations.
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.bar(df_durations["Track"], df_durations["Duration (s)"], color="b", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Track")
    plt.ylabel("Duration (seconds)")
    plt.title("Audio Track Durations")
    plt.show()


def plot_frequencies(df_frequencies):
    """
    Plot the frequency ranges of audio tracks.
    Parameters:
        df_frequencies (DataFrame): DataFrame containing track names and their frequency ranges.
    Returns:
        None
    """

    plt.figure(figsize=(12, 6))
    for i in range(len(df_frequencies)):
        plt.plot(
            [df_frequencies["Track"][i], df_frequencies["Track"][i]],
            [
                df_frequencies["Min Frequency (Hz)"][i],
                df_frequencies["Max Frequency (Hz)"][i],
            ],
            marker="o",
            color="r",
            linestyle="-",
            alpha=0.7,
        )

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Track")
    plt.ylabel("Frequency (Hz)")
    plt.title("Audio Frequency Ranges (Min to Max)")
    plt.show()


if __name__ == "__main__":
    base_path = "data/raw"
    data = {"TrackName": [], "Min Frequency": [], "Max Frequency": [], "Duration": []}

    for i in range(1, 21):  # Tracks are numbered from 1 to 20
        track_name = f"Track{i:05d}"
        wav_file = Path(base_path) / track_name / "mix.wav"

        if wav_file.exists():
            try:
                # Load audio file
                y, sr = librosa.load(wav_file, sr=None)  # Load at original sample rate
                duration = get_audio_duration(wav_file)
                min_freq, max_freq = get_frequency_range(y, sr)

                # Store frequency data separately
                data["TrackName"].append(track_name)
                data["Min Frequency"].append(min_freq)
                data["Max Frequency"].append(max_freq)
                data["Duration"].append(duration)

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
    save_to_summary(data)

    df = pd.DataFrame(data)

    # Compute averages
    avg_min_freq = np.mean(df["Min Frequency"]) if not df.empty else 0
    avg_max_freq = np.mean(df["Max Frequency"]) if not df.empty else 0
    avg_duration = np.mean(df["Duration"]) if not df.empty else 0

    # Print results
    print(f"\nAverage Minimum Frequency: {avg_min_freq:.2f} Hz")
    print(f"Average Maximum Frequency: {avg_max_freq:.2f} Hz")
    print(f"Average Track Duration: {avg_duration:.2f} seconds")
    # Convert to DataFrames and display separately
    # df_frequencies = pd.DataFrame(frequency_data)
    # df_durations = pd.DataFrame(duration_data)
    # save_info(duration_data, frequency_data)

    # Call the functions to generate plots
    # plot_durations(df_durations)
    # plot_frequencies(df_frequencies)

    # print("Audio Frequency Ranges:")
    # print(df_frequencies)

    # print("\nAudio Durations:")
    # print(df_durations)