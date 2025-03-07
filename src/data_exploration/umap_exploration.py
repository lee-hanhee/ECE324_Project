import numpy as np
import librosa
import umap.umap_ as umap  # Use umap-learn directly
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.io import wavfile




# Load all WAV files and extract raw waveform embeddings
base_path = "data/raw"
waveform_embeddings = []
track_labels = []

for i in range(1, 21):  # Tracks are numbered from 1 to 20
    track_name = f"Track{i:05d}"
    wav_file = Path(base_path) / track_name / "mix.wav"
    
    try:
        # Load waveform
        y, sr = librosa.load(str(wav_file), sr=22050)  # Downsample to 22.05kHz
        y = librosa.util.fix_length(y, size=22050*3)   # Fix length to 3 sec (optional)
        waveform_embeddings.append(y)
        track_labels.append(track_name)
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")

# Convert to NumPy array
waveform_embeddings = np.array(waveform_embeddings)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
waveform_umap = reducer.fit_transform(waveform_embeddings)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=waveform_umap[:, 0], y=waveform_umap[:, 1], hue=track_labels, palette="viridis")
plt.title("UMAP Projection of Waveform Data")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(loc="best")
plt.show()
