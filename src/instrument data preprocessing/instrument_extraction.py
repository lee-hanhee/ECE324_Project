import os
import yaml
import shutil
from glob import glob
from collections import defaultdict

# === SETUP ===

# Base directories
BASE_DIR = "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")
INSTRUMENTS_DIR = os.path.join(BASE_DIR, "instruments")

# Create instruments directory if it doesn't already exist
os.makedirs(INSTRUMENTS_DIR, exist_ok=True)

# Get all track directories matching pattern Track*
track_dirs = sorted(glob(os.path.join(RAW_DIR, "Track*")))

if not track_dirs:
    print("No track directories found! Check your path.")
else:
    print(f"Found {len(track_dirs)} track directories.")

# Dictionary to track how many files have been saved per instrument
instrument_counters = defaultdict(int)


# === MAIN LOOP: Process each track ===

for track_dir in track_dirs:
    metadata_path = os.path.join(track_dir, "metadata.yaml")

    # Skip track if no metadata found
    if not os.path.exists(metadata_path):
        print(f"Metadata not found for {track_dir}, skipping.")
        continue

    print(f"Processing: {track_dir}")

    # Load metadata YAML file
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    # Get stem dictionary (e.g., {'S01': {...}, 'S02': {...}, ...})
    stems = metadata.get("stems", {})

    # Folder containing audio stem files
    stem_dir = os.path.join(track_dir, "stems")

    # === PROCESS EACH STEM ===

    for stem_key, stem_info in stems.items():
        # Get instrument class label for this stem
        inst_class = stem_info.get("inst_class")

        if not inst_class:
            print(f"No inst_class for {stem_key} in {track_dir}")
            continue

        # Standardize instrument name to lowercase and clean special cases
        inst_class_lower = inst_class.lower()

        # Special case: handle "strings (continued)" as "strings"
        if inst_class_lower == "strings (continued)":
            inst_class_lower = "strings"

        # Construct source WAV file path
        wav_filename = f"{stem_key}.wav"
        wav_src_path = os.path.join(stem_dir, wav_filename)

        # Skip if the WAV file does not exist
        if not os.path.exists(wav_src_path):
            print(f"Missing WAV file: {wav_src_path}")
            continue

        # Create output folder for this instrument if it doesn't exist
        inst_folder = os.path.join(INSTRUMENTS_DIR, inst_class_lower)
        os.makedirs(inst_folder, exist_ok=True)

        # Use the counter to assign a unique filename
        file_index = instrument_counters[inst_class_lower]
        dest_filename = f"{inst_class_lower}_{file_index}.wav"
        dest_path = os.path.join(inst_folder, dest_filename)

        # Copy WAV file to instrument folder and update counter
        shutil.copy2(wav_src_path, dest_path)
        instrument_counters[inst_class_lower] += 1

        print(f"Copied {wav_filename} -> {inst_class_lower}/{dest_filename}")

# === DONE ===
print("Done.")
