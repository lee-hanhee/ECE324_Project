import numpy as np
import librosa
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import yaml  # To read metadata YAML files
from pathlib import Path
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder

# Base path for raw data
base_path = Path("data/raw")

# Store embeddings and labels
waveform_embeddings = []
instrument_labels = []  # Store instrument class dynamically

# Loop through each track folder
for track_folder in sorted(base_path.glob("Track*")):
    stems_folder = track_folder / "stems"  # ✅ Corrected stems folder path
    metadata_file = track_folder / "metadata.yaml"  # ✅ Metadata is at the same level as stems

    # Debugging print to check correct paths
    print(f"Checking: {metadata_file}")

    if not stems_folder.exists() or not metadata_file.exists():
        print(f"❌ Skipping {track_folder}: Metadata file or stems folder not found.")
        continue  # Skip if no stems or metadata file

    # Load metadata YAML
    with open(metadata_file, "r") as file:
        metadata = yaml.safe_load(file)

    # Process each stem file dynamically
    for stem_file in sorted(stems_folder.glob("S*.wav")):  # ✅ Now correctly searches inside stems/
        stem_name = stem_file.stem  # Extract "S00", "S01", etc.

        try:
            # Get instrument class from metadata
            instrument_class = metadata["stems"].get(stem_name, {}).get("inst_class", "Unknown")

            # Load waveform
            y, sr = librosa.load(str(stem_file), sr=22050)  # Downsample to 22.05kHz
            if len(y) == 0:
                raise ValueError("Empty audio file")
            y = librosa.util.fix_length(y, size=22050 * 3)  # Fix length to 3 sec

            # Append data
            waveform_embeddings.append(y)
            instrument_labels.append(instrument_class)  # Dynamically retrieved instrument class

        except Exception as e:
            print(f"Skipping {stem_file}: {e}")

# Convert to NumPy array
waveform_embeddings = np.array(waveform_embeddings)

# Encode instrument labels
instrument_encoder = LabelEncoder()
instrument_labels_encoded = instrument_encoder.fit_transform(instrument_labels)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
waveform_umap = reducer.fit_transform(waveform_embeddings)

# Plot UMAP results
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=waveform_umap[:, 0],
    y=waveform_umap[:, 1],
    hue=instrument_labels,  # Use dynamically extracted instrument labels
    palette="tab10",
    legend="full"
)

# Improve legend readability
plt.legend(title="Instrument Class", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("UMAP Projection of Individual Stems (Instruments)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.show()
