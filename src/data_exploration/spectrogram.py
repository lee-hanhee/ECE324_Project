import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
import librosa
import librosa.display
import seaborn as sns
import pandas as pd
from jaxtyping import Float, Array


def save_info(
    wav_data: Float[Array, "samples"], sample_rate: int
) -> tuple[Float[Array, "samples"], int]:
    """
    Save the sample rate and duration of the audio file.
    """
    duration = len(wav_data) / sample_rate
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total duration: {duration:.2f}s")
    print(f"Size of the input: {len(wav_data)}")
    return wav_data, sample_rate


def plot_spectrograms(
    mix_wav_data: Float[Array, "samples"],
    instrument_wav_datas: dict[str, Float[Array, "samples"]],
    sample_rate: int,
    track_name: str,
) -> str:
    """
    Plot the spectrograms of the mix and instrument audio files.
    Parameters:
      mix_wav_data (ndarray): Mixed audio data.
      instrument_wav_datas (dict): Dictionary of instrument audio data.
      sample_rate (int): Sample rate of the audio data.
      track_name (str): Name of the track.
    Returns:
      str: Path to the saved spectrogram image.
    """
    num_instruments = len(instrument_wav_datas)
    total_plots = num_instruments + 1

    # Create a figure for the spectrograms
    plt.figure(figsize=(12, 3 * total_plots))
    plt.subplot(total_plots, 1, 1)
    plt.specgram(mix_wav_data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis")
    plt.title("Spectrogram of mix.wav")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")

    # plot spectrograms for each instrument
    for i, (instrument_name, instrument_wav) in enumerate(instrument_wav_datas.items()):
        plt.subplot(total_plots, 1, i + 2)
        plt.specgram(
            instrument_wav, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis"
        )
        plt.title(f"Spectrogram of {instrument_name}.wav")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Intensity (dB)")

    plt.tight_layout()
    image_path = f"results/spectrograms/{track_name}-spectrogram.png"
    plt.savefig(image_path)
    plt.close()

    return image_path


def plot_waveform(
    mix_y: Float[Array, "samples"],
    mix_sr: int,
    mix_track_name: str,
    stems_y: list[Float[Array, "samples"]],
    stems_sr: list[int],
) -> None:
    """
    Plot the waveform of the mixed audio and its stems.
    Parameters:
        mix_y (ndarray): Mixed audio data.
        mix_sr (int): Sample rate of the mixed audio.
        mix_track_name (str): Name of the mixed track.
        stems_y (list): List of stem audio data.
        stems_sr (list): List of sample rates for each stem.
    Returns:
        None
    """
    n_rows = len(stems_y) + 1
    fig, ax = plt.subplots(nrows=n_rows, ncols=1, sharex=True, figsize=(10, 8))
    for i, (y, sr) in enumerate(zip(stems_y, stems_sr)):
        librosa.display.waveshow(y, sr=sr, ax=ax[i], label=f"Source {i}", axis="time")
        ax[i].set(xlabel="Time [s]")
        ax[i].legend()

    librosa.display.waveshow(
        mix_y, sr=mix_sr, ax=ax[-1], label="Mixed Audio", axis="time"
    )
    ax[-1].set(xlabel="Time [s]")
    ax[-1].legend()

    # Set the overall title
    fig.suptitle("Waveform - " + mix_track_name)

    # Save the figure
    image_path = f"results/waveforms/{mix_track_name}-waveforms.png"
    fig.savefig(image_path)
    plt.close(fig)


def calculate_rms(audio: Float[Array, "samples"]) -> float:
    """
    Calculate the Root Mean Square (RMS) of the audio signal.
    """
    return np.sqrt(np.mean(np.square(audio)))


def plot_boxplot_histogram(rms_values):
    """
    Plot a histogram and boxplot of the RMS values.
    Parameters:
        rms_values (list): List of RMS values.
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.hist(rms_values, bins=20, color="skyblue", edgecolor="black")
    plt.title("Histogram of RMS (Volume) Levels")
    plt.xlabel("RMS (Volume Level)")
    plt.ylabel("Frequency")
    image_path = "results/volume_boxplot.png"
    plt.savefig(image_path)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=rms_values, color="skyblue")
    plt.title("Boxplot of RMS (Volume) Levels")
    plt.ylabel("RMS (Volume Level)")
    image_path = "results/volume_histogram.png"
    plt.savefig(image_path)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("results/metrics/data_summary.csv")

    base_path = "data/raw"
    rms = []
    instrument_data = {}

    for i in range(1, 21):  # Tracks are numbered from 1 to 20
        track_name = f"Track{i:05d}"
        wav_file = Path(base_path) / track_name / "mix.wav"
        sample_rate, wav_mix_data = wavfile.read(wav_file, "rb")
        y_mix, sr_mix = librosa.load(wav_file)
        rms.append(calculate_rms(y_mix))
        y_stems = []
        sr_stems = []
        for inst in range(df["Number of Instruments"][i - 1]):
            stem_name = f"S{inst:02d}.wav"
            stem = f"S{inst:02d}"
            inst_file = Path(base_path) / track_name / Path("stems") / stem_name
            # print(inst_file)
            try:
                _, wav_inst_data = wavfile.read(inst_file, "rb")
                y, sr = librosa.load(inst_file)
                y_stems.append(y)
                sr_stems.append(sr)
                instrument_data[stem] = wav_inst_data
            except:
                continue

        # plot_spectrograms(wav_mix_data, instrument_data, sample_rate, track_name)
        plot_waveform(y_mix, sr_mix, track_name, y_stems, sr_stems)
        print("Done " + track_name)

    plot_boxplot_histogram(rms)