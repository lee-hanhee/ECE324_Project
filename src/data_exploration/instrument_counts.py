import os
import yaml
import csv
import matplotlib.pyplot as plt
from collections import Counter

# Define the path to the directory containing the tracks
base_path = "data/raw"

def extract_instrument_data(base_path):
    '''Extracts instrument data from metadata files in the specified directory.
    Parameters:
        base_path (str): The path to the directory containing the track folders.
    Returns:
        tuple: A tuple containing two dictionaries:
            - instrument_distributions: A dictionary with instrument distributions for each track.
            - instrument_counts: A dictionary with the count of instruments for each track.
    '''
    # Dictionary to store instrument distributions for each track
    instrument_distributions = {}
    instrument_counts = {}

    # Loop through all tracks
    for i in range(1, 21):  # Tracks are numbered from 1 to 20
        track_folder = os.path.join(base_path, f"Track{str(i).zfill(5)}")
        metadata_file = os.path.join(track_folder, "metadata.yaml")

        if not os.path.exists(metadata_file):
            print(f"Metadata file not found for Track{str(i).zfill(5)}")
            continue

        # Read the metadata file
        with open(metadata_file, "r") as file:
            metadata = yaml.safe_load(file)

        # Extract instruments from the stems section
        stems = metadata.get("stems", {})
        inst_classes = [stem_data["inst_class"] for stem_data in stems.values() if "inst_class" in stem_data]
        if "Sound effects" in inst_classes:
            inst_classes.remove("Sound effects")
            inst_classes.append("Sound Effects")

        # Count the number of instruments and their distribution
        track_key = f"Track{str(i).zfill(5)}"
        instrument_distributions[track_key] = Counter(inst_classes)
        instrument_counts[track_key] = len(inst_classes)

    return instrument_distributions, instrument_counts

def plot_combined_instrument_distribution(instrument_distributions):
    # Aggregate counts across all tracks
    total_counts = Counter()
    for dist in instrument_distributions.values():
        total_counts.update(dist)

    # Sort the total counts by value (highest to lowest)
    sorted_counts = dict(total_counts.most_common())

    # Plot single combined bar graph
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_counts.keys(), sorted_counts.values())
    plt.title("Overall Instrument Class Distribution Across All Tracks")
    plt.xlabel("Instrument Classes")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join("results", "plots", "inst_dist")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_tracks_inst_dist.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Saved combined instrument distribution plot at {output_file}")

def main():
    instrument_distributions, instrument_counts = extract_instrument_data(base_path)
    plot_combined_instrument_distribution(instrument_distributions)

if __name__ == "__main__":
    main()
