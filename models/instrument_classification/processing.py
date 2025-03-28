import numpy as np
import os


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
    return a one-hot vector of size 14.
    """
    name = os.path.splitext(filename)[0]  # Remove .wav
    instrument_names = name.split("_")[:-1]  # Remove numeric index
    
    # Some instruments have multiple words (e.g., "synth lead")
    # Rebuild full instrument names from fragments
    instrument_names = "_".join(instrument_names).split("_")
    
    # Handle multi-word instruments properly (e.g. 'synth lead', 'chromatic percussion')
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

    # Create one-hot vector
    one_hot = np.zeros(len(INSTRUMENT_CLASSES), dtype=int)
    for inst in resolved:
        if inst in INST_TO_INDEX:
            one_hot[INST_TO_INDEX[inst]] = 1
        else:
            print(f"Warning: '{inst}' not found in instrument list.")

    return one_hot

