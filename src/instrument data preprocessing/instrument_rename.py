import os

# Set the base instruments folder
INSTRUMENTS_DIR = "data/instruments"

# Loop through each instrument subfolder
for instrument in os.listdir(INSTRUMENTS_DIR):
    instrument_path = os.path.join(INSTRUMENTS_DIR, instrument)
    if not os.path.isdir(instrument_path):
        continue

    # Get all .wav files in this instrument folder
    wav_files = sorted([f for f in os.listdir(instrument_path) if f.endswith(".wav")])

    # Rename files to lowercase instrument name: instrument_0.wav, instrument_1.wav, ...
    instrument_lower = instrument.lower()

    for i, filename in enumerate(wav_files):
        src = os.path.join(instrument_path, filename)
        dst = os.path.join(instrument_path, f"{instrument_lower}_{i}.wav")
        os.rename(src, dst)
        print(f"Renamed {filename} -> {instrument_lower}_{i}.wav")

print("Renaming complete.")
