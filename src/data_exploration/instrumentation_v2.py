import os
import yaml
import csv
import matplotlib.pyplot as plt
from collections import Counter

# Define the path to the directory containing the tracks
# base_path = "../../data/raw"
base_path = "data/raw"

def extract_instrument_data(base_path):
    # Dictionary to store instrument distributions for each track
    instrument_distributions = {}
    instrument_counts = {}

    # Loop through all tracks
    for i in range(1, 21):  # Tracks are numbered from 1 to 20
        if i < 10:
            track_folder = os.path.join(base_path, f"Track0000{i}")
        else:
            track_folder = os.path.join(base_path, f"Track000{i}")
        metadata_file = os.path.join(track_folder, "metadata.yaml")

        if not os.path.exists(metadata_file):
            print(f"Metadata file not found for Track0000{i}" if i < 10 else f"Metadata file not found for Track000{i}")
            continue

        # Read the metadata file
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)

        # Extract instruments from the stems section
        stems = metadata.get("stems", {})
        inst_classes = [stem_data["inst_class"] for stem_data in stems.values() if "inst_class" in stem_data]

        # Count the number of instruments and their distribution
        track_key = f"Track0000{i}" if i < 10 else f"Track000{i}"
        instrument_distributions[track_key] = Counter(inst_classes)
        instrument_counts[track_key] = len(inst_classes)

    return instrument_distributions, instrument_counts

def save_instrument_counts(instrument_counts):
    # output_file = os.path.join("../../results/metrics", "data_summary.csv")
    # os.makedirs("../../results/metrics", exist_ok=True)
    output_file = os.path.join("results/metrics", "data_summary.csv")
    os.makedirs("results/metrics", exist_ok=True)

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Track", "Instrument Count"])  # Header row
        for track, count in instrument_counts.items():
            writer.writerow([track, count])

    print(f"Instrument counts saved at {output_file}")

def plot_instrument_distributions(instrument_distributions):
    for track, distribution in instrument_distributions.items():
        # Create a bar graph for the current track
        plt.figure(figsize=(10, 6))
        plt.bar(distribution.keys(), distribution.values())
        plt.title(f"Instrument Distribution for {track}")
        plt.xlabel("Instrument Classes")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot as an image
        output_file = os.path.join("../../results/plots/inst_dist", f"{track}_inst_dist.png")
        os.makedirs("../../results/plots/inst_dist", exist_ok=True)
        plt.savefig(output_file)
        plt.close()

        print(f"Saved bar graph for {track} at {output_file}")

def main():
    instrument_distributions, instrument_counts = extract_instrument_data(base_path)
    plot_instrument_distributions(instrument_distributions)
    save_instrument_counts(instrument_counts)

if __name__ == "__main__":
    main()
    
'''
PROMPT for ChatGPT:
I have data in data/raw/Track0000<Number>/metadata.yaml, which contains the metadata for the instruments and number of instruments playing in each .wav file, 
can you write a python script that extracts the number of instruments and distribution of instruments (use bar graph) for each Track0000<Number>, 
there're 20 tracks, and here is an example of what the metadata file looks like: 
UUID: 136e224ff9848e53c33322f58d58f1b3
audio_dir: stems
lmd_midi_dir: lmd_matched/Z/V/A/TRZVABW12903CC606A/136e224ff9848e53c33322f58d58f1b3.mid
midi_dir: MIDI
normalization_factor: -13.0
normalized: true
overall_gain: 0.18308985769022404
stems:
  S00:
    audio_rendered: false
    inst_class: Piano
    integrated_loudness: -27.747527318151928
    is_drum: false
    midi_program_name: Bright Acoustic Piano
    midi_saved: false
    plugin_name: the_grandeur.nkm
    program_num: 1
  S01:
    audio_rendered: false
    inst_class: Bass
    integrated_loudness: -27.747503054929144
    is_drum: false
    midi_program_name: Electric Bass (pick)
    midi_saved: false
    plugin_name: classic_bass.nkm
    program_num: 34
  S02:
    audio_rendered: false
    inst_class: Guitar
    integrated_loudness: -27.74756478669523
    is_drum: false
    midi_program_name: Acoustic Guitar (steel)
    midi_saved: false
    plugin_name: nylon_guitar.nkm
    program_num: 25
  S03:
    audio_rendered: false
    inst_class: Guitar
    integrated_loudness: -27.749945880219656
    is_drum: false
    midi_program_name: Overdriven Guitar
    midi_saved: false
    plugin_name: elektrik_guitar.nkm
    program_num: 29
  S04:
    audio_rendered: false
    inst_class: Guitar
    integrated_loudness: -27.748257487948777
    is_drum: false
    midi_program_name: Electric Guitar (jazz)
    midi_saved: false
    plugin_name: funk_guitar.nkm
    program_num: 26
  S05:
    audio_rendered: false
    inst_class: Ethnic
    is_drum: false
    midi_program_name: Fiddle
    midi_saved: false
    plugin_name: None
    program_num: 110
  S06:
    audio_rendered: false
    inst_class: Drums
    integrated_loudness: -27.7476048963183
    is_drum: true
    midi_program_name: Drums
    midi_saved: false
    plugin_name: pop_kit.nkm
    program_num: 128
  S07:
    audio_rendered: false
    inst_class: Guitar
    integrated_loudness: -27.747390490033656
    is_drum: false
    midi_program_name: Electric Guitar (clean)
    midi_saved: false
    plugin_name: jazz_guitar2.nkm
    program_num: 27
  S08:
    audio_rendered: false
    inst_class: Guitar
    integrated_loudness: -27.74736790802703
    is_drum: false
    midi_program_name: Acoustic Guitar (steel)
    midi_saved: false
    plugin_name: nylon_guitar.nkm
    program_num: 25
target_peak: -1.0, 

Extract the number of instruments from the last number of S0# and create a bar graph distribution from the inst_class. Use Python.
'''