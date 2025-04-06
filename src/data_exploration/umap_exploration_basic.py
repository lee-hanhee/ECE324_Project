import numpy as np
import librosa
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import yaml  # To read metadata YAML files
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Base path for raw data
base_path = Path("data/raw")

# Containers for waveform embeddings and corresponding instrument labels
waveform_embeddings = []
instrument_labels = []

# Loop through each track folder (e.g., Track001, Track002, etc.)
for track_folder in sorted(base_path.glob("Track*")):
    stems_folder = track_folder / "stems"
    metadata_file = track_folder / "metadata.yaml"

    track_name = track_folder.name  # Extract track folder name

    print(f"\nProcessing: {track_name}")

    # Skip the track if either the stems folder or metadata file is missing
    if not stems_folder.exists() or not metadata_file.exists():
        print(f"Skipping {track_name}: Metadata file or stems folder not found.")
        continue

    # Load YAML metadata for the current track
    with open(metadata_file, "r") as file:
        metadata = yaml.safe_load(file)

    # If 'stems' key is missing in metadata, skip the track
    if "stems" not in metadata:
        print(f"WARNING: No 'stems' key in metadata for {track_name}")
        continue

    # Keep track of unique instrument classes to avoid duplicates per track
    unique_instruments = set()

    # Process each stem WAV file in the current track
    for stem_file in sorted(stems_folder.glob("S*.wav")):
        stem_name = stem_file.stem  # Get file name without extension

        try:
            # Extract instrument class from metadata, default to "Unknown"
            instrument_class = metadata["stems"].get(
                stem_name, {}
            ).get("inst_class", "Unknown")

            # Skip if this instrument class has already been processed for the track
            if instrument_class in unique_instruments:
                continue

            unique_instruments.add(instrument_class)

            # Load audio waveform and downsample to 120000 Hz
            y, sr = librosa.load(str(stem_file), sr=120000)
            if len(y) == 0:
                raise ValueError("Empty audio file")

            # Fix waveform length to 3 seconds (3 * 22050 samples)
            y = librosa.util.fix_length(y, size=22050 * 3)

            # Append waveform and label to lists
            waveform_embeddings.append(y)
            instrument_labels.append(instrument_class)

        except Exception as e:
            print(f"  Skipping {stem_file}: {e}")

# Convert list of waveforms to NumPy array
waveform_embeddings = np.array(waveform_embeddings)

# Debug info: number of unique instrument classes processed
print(f"\nTotal unique instrument classes processed: {len(set(instrument_labels))}")

# Encode instrument labels to integers
instrument_encoder = LabelEncoder()
instrument_labels_encoded = instrument_encoder.fit_transform(instrument_labels)

# Apply UMAP for dimensionality reduction (to 2D)
reducer = umap.UMAP(n_components=2, random_state=42)
waveform_umap = reducer.fit_transform(waveform_embeddings)

# Plot UMAP projection
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    x=waveform_umap[:, 0],
    y=waveform_umap[:, 1],
    hue=instrument_labels,
    palette="tab10",
    legend="full"
)

# Improve legend readability
plt.legend(title="Instrument Class", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("UMAP Projection of Individual Stems (Unique Instrument Classes)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.tight_layout()  # Adjust layout to fit elements within the figure

# Save figure to results/plots/umap
output_path = Path("results/plots/umap/umap_projection_basic.png")
output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

plt.savefig(output_path, dpi=300)
print(f"UMAP plot saved to: {output_path}")

plt.close()
