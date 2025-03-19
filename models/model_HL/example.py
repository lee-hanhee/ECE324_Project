#!/usr/bin/env python3
"""
Example script demonstrating the complete workflow:
1. Train the instrument classifier (if models don't exist)
2. Train the separation models (if models don't exist)
3. Extract instruments from an audio file
"""

import os
import argparse
import torch
from pathlib import Path

def check_models_exist(model_dir, model_type='classifier'):
    """Check if models exist in the given directory."""
    if model_type == 'classifier':
        # Check ensemble directory
        ensemble_dir = os.path.join(model_dir, 'ensemble')
        if os.path.exists(ensemble_dir) and os.path.isdir(ensemble_dir):
            models = [f for f in os.listdir(ensemble_dir) if f.endswith('.pth')]
            if models:
                return True
                
        # Check standard locations
        models = [f for f in os.listdir(model_dir) if f.endswith('.pth') and 'ensemble_model' in f]
        return len(models) > 0
        
    elif model_type == 'separator':
        # Check separation_models directory
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('_separator.pth')]
            return len(models) > 0
            
    return False

def main():
    parser = argparse.ArgumentParser(description='Instrument Classification and Separation Example')
    parser.add_argument('--input', type=str, help='Path to input audio file (required)')
    parser.add_argument('--output_dir', type=str, default='./extracted_instruments',
                        help='Directory to save extracted instruments')
    parser.add_argument('--train_if_needed', action='store_true',
                        help='Train models if they do not exist')
    parser.add_argument('--format', type=str, choices=['wav', 'mp3'], default='wav',
                        help='Output audio format')
    args = parser.parse_args()
    
    if not args.input:
        parser.error('Please specify an input audio file with --input')
    
    # Set up directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_dir = os.path.join(script_dir)
    separation_dir = os.path.join(script_dir, 'separation_models')
    
    # Make sure separation_models directory exists
    os.makedirs(separation_dir, exist_ok=True)
    
    # Step 1: Check if classifier models exist
    classifier_exists = check_models_exist(classifier_dir, 'classifier')
    if not classifier_exists:
        if args.train_if_needed:
            print("No classifier models found. Training classifier...")
            import train
            train.main()
        else:
            print("No classifier models found. Please train the classifier first with:")
            print("python train.py")
            return
    
    # Step 2: Check if separation models exist
    separator_exists = check_models_exist(separation_dir, 'separator')
    if not separator_exists:
        if args.train_if_needed:
            print("No separation models found. Training separation models...")
            import train_separation
            # Run the train_separation script with appropriate arguments
            import sys
            sys.argv = [
                'train_separation.py',
                '--classifier_dir', classifier_dir,
                '--output_dir', separation_dir,
                # You can add more arguments here
            ]
            train_separation.main()
        else:
            print("No separation models found. Please train separation models first with:")
            print(f"python train_separation.py --classifier_dir {classifier_dir} --output_dir {separation_dir}")
            return
    
    # Step 3: Extract instruments from the audio file
    print(f"Extracting instruments from {args.input}...")
    import extract_instruments
    
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract instruments
    extract_instruments.extract_instruments(
        input_path=args.input,
        output_dir=args.output_dir,
        classifier_dir=classifier_dir,
        separation_dir=separation_dir,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        output_format=args.format
    )
    
    print(f"\nAll done! Extracted instruments saved to {args.output_dir}")
    print(f"Visualization of waveforms: {os.path.join(args.output_dir, 'waveform_comparison.png')}")
    print(f"Visualization of spectrograms: {os.path.join(args.output_dir, 'spectrogram_comparison.png')}")

if __name__ == "__main__":
    main() 