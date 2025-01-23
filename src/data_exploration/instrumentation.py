import librosa

# Load the WAV file (returns waveform as a NumPy array and sampling rate)
audio, sr = librosa.load('file.wav', sr=None)

# Print details
print(f"Sampling Rate: {sr}")
print(f"Audio Shape: {audio.shape}")

# Extract features (e.g., MFCCs)
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print(f"MFCCs Shape: {mfccs.shape}")
