import numpy as np
import os
import random

# Define all instruments in the fixed order
INSTRUMENT_CLASSES = [
    'bass',
    'brass',
    'chromatic percussion',
    'drums',
    'guitar',
    'organ',
    'piano',
    'pipe',
    'reed',
    'strings',
    'strings (continued)',
    'synth lead',
    'synth pad'
]

# Mapping from name to index
INST_TO_INDEX = {inst: idx for idx, inst in enumerate(INSTRUMENT_CLASSES)}

def get_one_hot_label_from_filename(filename):
    """
    Given a filename like 'piano_0.wav' or 'drums_guitar_1.wav',
    return a one-hot vector of size 13 and also return index list.
    """
    name = os.path.splitext(filename)[0]  # Remove .wav
    instrument_names = name.split("_")[:-1]  # Remove numeric index
    
    # Recombine and split to handle multi-word instruments
    instrument_names = "_".join(instrument_names).split("_")
    
    resolved = []
    i = 0
    while i < len(instrument_names):
        current = instrument_names[i]
        if i + 1 < len(instrument_names):
            two_word = current + " " + instrument_names[i + 1]
            if two_word in INST_TO_INDEX:
                resolved.append(two_word)
                i += 2
                continue
        resolved.append(current)
        i += 1

    # Get index list (not one-hot)
    index_list = []
    for inst in resolved:
        if inst in INST_TO_INDEX:
            index_list.append(INST_TO_INDEX[inst])
        else:
            print(f"Warning: '{inst}' not found in instrument list.")

    return index_list

def get_data(percent=1.0, seed=42):
    """
    Get a representative subset of the dataset.
    
    Parameters:
        percent (float): A float between 0 and 1, percentage of data to keep from each class.
        seed (int): Random seed for reproducibility.
    
    Returns:
        stem_dict (dict): Maps file paths to a list of label indices.
    """
    random.seed(seed)
    base_dir = "data/instruments"
    stem_dict = {}

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Collect all .wav files in the folder
        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        wav_paths = [os.path.abspath(os.path.join(folder_path, f)) for f in wav_files]

        if not wav_paths:
            continue

        # Select a percentage of them
        k = max(1, int(len(wav_paths) * percent))  # At least 1 per class
        selected = random.sample(wav_paths, k)

        for path in selected:
            label_indices = get_one_hot_label_from_filename(os.path.basename(path))
            stem_dict[path] = label_indices

    return stem_dict

# Get 25% of the data from each class
data = get_data(percent=0.1)

# Print sample
for path, labels in list(data.items()):
    print(path, "->", labels)