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

# Store embeddings and labels
waveform_embeddings = []
instrument_labels = []

# Loop through each track folder
for track_folder in sorted(base_path.glob("Track*")):
    stems_folder = track_folder / "stems"
    metadata_file = track_folder / "metadata.yaml"

    track_name = track_folder.name  # Extract TrackXXXX name

    print(f"\nProcessing: {track_name}")

    if not stems_folder.exists() or not metadata_file.exists():
        print(f"Skipping {track_name}: Metadata file or stems folder not found.")
        continue  # Skip if no stems or metadata file

    # Load metadata YAML
    with open(metadata_file, "r") as file:
        metadata = yaml.safe_load(file)

    if "stems" not in metadata:
        print(f"WARNING: No 'stems' key in metadata for {track_name}")
        continue

    # Track unique instrument classes per track
    unique_instruments = set()

    # Process each stem file
    for stem_file in sorted(stems_folder.glob("S*.wav")):
        stem_name = stem_file.stem  # Extract filename without .wav

        try:
            # Retrieve instrument class from metadata
            instrument_class = metadata["stems"].get(stem_name, {}).get("inst_class", "Unknown")

            # If this instrument class was already processed, skip it
            if instrument_class in unique_instruments:
                continue  # Ensures only unique `inst_class` are included

            unique_instruments.add(instrument_class)  # Track unique instrument classes

            # Load waveform
            y, sr = librosa.load(str(stem_file), sr=120000)  # Downsample to 22.05kHz
            if len(y) == 0:
                raise ValueError("Empty audio file")
            y = librosa.util.fix_length(y, size=22050 * 3)  # Fix length to 3 sec

            # Append data
            waveform_embeddings.append(y)
            instrument_labels.append(instrument_class)

        except Exception as e:
            print(f"  Skipping {stem_file}: {e}")

# Convert to NumPy array
waveform_embeddings = np.array(waveform_embeddings)

# Debug: Print total unique instruments processed
print(f"\nTotal unique instrument classes processed: {len(set(instrument_labels))}")

# Encode instrument labels
instrument_encoder = LabelEncoder()
instrument_labels_encoded = instrument_encoder.fit_transform(instrument_labels)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
waveform_umap = reducer.fit_transform(waveform_embeddings)

# Plot UMAP results
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

plt.show()
