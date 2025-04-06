import os
import itertools
import random
import librosa
import soundfile as sf

# Base directory for instrument files
INSTRUMENTS_DIR = "data/instruments"
OUTPUT_DIR = os.path.join(INSTRUMENTS_DIR, "combined")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Instruments to combine
instrument_list = ["Piano", "Drums", "Guitar", "Bass", "Strings"]

# Number of mixes to generate per instrument combination
mixes_per_combination = 5  # You can increase this

# Load all .wav file paths from instrument folders
instrument_files = {}
for inst in instrument_list:
    inst_dir = os.path.join(INSTRUMENTS_DIR, inst)
    wavs = sorted([
        os.path.join(inst_dir, f)
        for f in os.listdir(inst_dir)
        if f.endswith(".wav")
    ])
    instrument_files[inst] = wavs

# Create combinations of 2, 3, and 4 instruments
for r in range(2, len(instrument_list) + 1):
    for combo in itertools.combinations(instrument_list, r):
        combo_name = "_".join(inst.lower() for inst in combo)

        for i in range(mixes_per_combination):
            # Pick one random file per instrument in the combo
            selected_files = [
                random.choice(instrument_files[inst]) for inst in combo
            ]

            # Load and prepare signals
            signals = []
            min_length = float("inf")
            sr = None

            for file_path in selected_files:
                y, sr = librosa.load(file_path, sr=None)
                signals.append(y)
                min_length = min(min_length, len(y))

            # Trim all signals to match shortest signal length
            signals = [s[:min_length] for s in signals]

            # Mix by averaging (can modify to sum or use weights)
            mix = sum(signals) / len(signals)

            # Output filename and save mixed signal
            output_name = f"{combo_name}_{i}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            sf.write(output_path, mix, sr)
            print(f"Created: {output_name}")

print("All combinations generated.")
