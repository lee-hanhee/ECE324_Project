import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter

# Define the path to the directory containing the tracks
base_path = "../../data/raw"

def extract_instrument_data(base_path):
    # Dictionary to store instrument distributions for each track
    instrument_distributions = {}

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
        instrument_distributions[f"Track0000{i}" if i < 10 else f"Track000{i}"] = Counter(inst_classes)

    return instrument_distributions

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
    instrument_distributions = extract_instrument_data(base_path)
    plot_instrument_distributions(instrument_distributions)

if __name__ == "__main__":
    main()