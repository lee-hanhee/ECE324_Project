import os
import yaml
import shutil
from glob import glob
from collections import defaultdict

# Base directories
BASE_DIR = "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")
INSTRUMENTS_DIR = os.path.join(BASE_DIR, "instruments")

# Ensure instruments folder exists
os.makedirs(INSTRUMENTS_DIR, exist_ok=True)

# Get all raw track folders like Track00001, Track00002, ...
track_dirs = sorted(glob(os.path.join(RAW_DIR, "Track*")))

if not track_dirs:
    print("No track directories found! Check your path.")
else:
    print(f"Found {len(track_dirs)} track directories.")

# Dictionary to keep track of how many files we've copied for each instrument
instrument_counters = defaultdict(int)

for track_dir in track_dirs:
    metadata_path = os.path.join(track_dir, "metadata.yaml")

    if not os.path.exists(metadata_path):
        print(f"Metadata not found for {track_dir}, skipping.")
        continue

    print(f"Processing: {track_dir}")

    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    stems = metadata.get("stems", {})
    stem_dir = os.path.join(track_dir, "stems")

    for stem_key, stem_info in stems.items():
        inst_class = stem_info.get("inst_class")
        if not inst_class:
            print(f"No inst_class for {stem_key} in {track_dir}")
            continue

        wav_filename = f"{stem_key}.wav"
        wav_src_path = os.path.join(stem_dir, wav_filename)

        if not os.path.exists(wav_src_path):
            print(f"Missing WAV file: {wav_src_path}")
            continue

        # Standardize instrument name to lowercase
        inst_class_lower = inst_class.lower()
        inst_folder = os.path.join(INSTRUMENTS_DIR, inst_class_lower)
        if inst_class_lower == "strings (continued)":
            # Special case for "strings (continued)" to avoid confusion
            inst_class_lower = "strings"
        os.makedirs(inst_folder, exist_ok=True)

        # Use the current count for naming
        file_index = instrument_counters[inst_class_lower]
        if inst_class_lower == "strings (continued)":
            # Special case for "strings (continued)" to avoid confusion
            inst_class_lower = "strings_"
        dest_filename = f"{inst_class_lower}_{file_index}.wav"
        dest_path = os.path.join(inst_folder, dest_filename)

        # Copy and update counter
        shutil.copy2(wav_src_path, dest_path)
        instrument_counters[inst_class_lower] += 1

        print(f"Copied {wav_filename} -> {inst_class_lower}/{dest_filename}")

print("Done.")
