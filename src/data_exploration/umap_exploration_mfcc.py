import numpy as np
import librosa
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Base path for raw data
base_path = Path("data/raw")

# Store feature embeddings and labels
feature_embeddings = []
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
            y, sr = librosa.load(str(stem_file), sr=22050)  # Downsample to 22.05kHz
            if len(y) == 0:
                raise ValueError("Empty audio file")
            y = librosa.util.fix_length(y, size=22050 * 3)  # Fix length to 3 sec

            # Extract Features
            # 1. MFCCs (40 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfccs)  # First derivative
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)  # Second derivative
            mfcc_features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
            mfcc_vector = np.mean(mfcc_features, axis=1)  # Shape: (120,)

            # 2. Chroma Features (Harmonic/Pitch)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_vector = np.mean(chroma, axis=1)  # Shape: (12,)

            # 3. Spectral Contrast (Frequency Band Differentiation)
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spec_vector = np.mean(spec_contrast, axis=1)  # Shape: (7,)

            # 4. Mel Spectrogram (Alternative to MFCC)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_vector = np.mean(mel_spec, axis=1)  # Shape: (128,)

            # Combine all extracted features
            final_vector = np.concatenate([mfcc_vector, chroma_vector, spec_vector, mel_vector])
            #final_vector = np.concatenate([mfcc_vector, chroma_vector])
            #final_vector = np.concatenate([spec_vector, mel_vector])

            # Append data
            feature_embeddings.append(final_vector)
            instrument_labels.append(instrument_class)

        except Exception as e:
            print(f"  Skipping {stem_file}: {e}")

# Convert to NumPy array
feature_embeddings = np.array(feature_embeddings)

# Normalize features using Z-score standardization
scaler = StandardScaler()
feature_embeddings = scaler.fit_transform(feature_embeddings)

# Debug: Print total unique instruments processed
print(f"\nTotal unique instrument classes processed: {len(set(instrument_labels))}")

# Encode instrument labels
instrument_encoder = LabelEncoder()
instrument_labels_encoded = instrument_encoder.fit_transform(instrument_labels)

pca = PCA(n_components=50)  # Reduce dimensionality before UMAP
pca_features = pca.fit_transform(feature_embeddings)

# Apply UMAP with optimized hyperparameters
reducer = umap.UMAP(n_components=2, n_neighbors=60, min_dist=0.4,metric="cosine", random_state=42)
umap_embeddings = reducer.fit_transform(feature_embeddings)

# Plot UMAP results
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    x=umap_embeddings[:, 0],
    y=umap_embeddings[:, 1],
    hue=instrument_labels,
    palette="tab10",
    legend="full"
)

# Improve legend readability
plt.legend(title="Instrument Class", bbox_to_anchor=(1.05, 1), loc='upper left')

# Increase font sizes
plt.title("UMAP Projection of Individual Stems - MFCC", fontsize=20)  # Bigger title
plt.xlabel("UMAP Component 1", fontsize=16)  # Bigger X-axis label
plt.ylabel("UMAP Component 2", fontsize=16)  # Bigger Y-axis label

# Increase tick font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()  # Adjust layout to fit elements within the figure

# Save figure to results/plots/umap
output_path = Path("results/plots/umap/umap_projection_mfcc.png")
output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

plt.savefig(output_path, dpi=300)
print(f"UMAP plot saved to: {output_path}")

plt.close()

