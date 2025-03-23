import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchaudio
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import torch.nn.functional as F

from dataset import InstrumentDataset
from separation import InstrumentSeparator, DemucsModel

def ensure_matching_shapes(output, target):
    """
    Ensure that output and target tensors have the same shape for loss computation.
    
    Args:
        output: Model output tensor
        target: Target tensor
        
    Returns:
        (output, target) with matching shapes
    """
    # Remove any singleton dimensions from target
    # Add a safety counter to prevent infinite loops
    safety_counter = 0
    max_squeezes = 5  # Reasonable limit to prevent hanging
    
    while len(target.shape) > 3 and safety_counter < max_squeezes:
        # Print shape before squeeze for debugging
        old_shape = target.shape
        target = target.squeeze(-1)
        # If shape didn't change, break to avoid infinite loop
        if target.shape == old_shape:
            break
        safety_counter += 1
    
    # Match batch dimension if needed
    if output.shape[0] != target.shape[0]:
        raise ValueError(f"Batch size mismatch: output {output.shape[0]} vs target {target.shape[0]}")
    
    # Match time dimension (last dimension)
    if output.shape[2] != target.shape[2]:
        min_length = min(output.shape[2], target.shape[2])
        output = output[..., :min_length]
        target = target[..., :min_length]
    
    return output, target

def compute_spectral_loss(output, target, device, window_size=2048, hop_length=512):
    """
    Compute spectral loss between output and target waveforms.
    
    Args:
        output: Output waveform tensor [batch, channels, time]
        target: Target waveform tensor [batch, channels, time]
        device: Device to perform computation on
        window_size: FFT window size
        hop_length: Hop length
        
    Returns:
        Spectral loss
    """
    # Ensure both tensors are 3D with shape [batch, channels, time]
    if len(output.shape) != 3:
        output = output.view(output.shape[0], 1, -1)
    if len(target.shape) != 3:
        target = target.view(target.shape[0], 1, -1)
    
    # Match time dimension if different
    if output.shape[2] != target.shape[2]:
        min_length = min(output.shape[2], target.shape[2])
        output = output[..., :min_length]
        target = target[..., :min_length]
    
    # Flatten channels into batch for STFT
    b, c, t = output.shape
    output_2d = output.reshape(b * c, -1)  # [batch*channels, time]
    target_2d = target.reshape(b * c, -1)  # [batch*channels, time]
    
    # Ensure length is sufficient for STFT
    min_required_length = hop_length * 4  # STFT needs at least a few windows
    if output_2d.shape[1] < min_required_length:
        pad_size = min_required_length - output_2d.shape[1]
        output_2d = F.pad(output_2d, (0, pad_size))
        target_2d = F.pad(target_2d, (0, pad_size))
    
    # Create Hann window
    window = torch.hann_window(window_size).to(device)
    
    try:
        # Compute STFT with return_complex=True for PyTorch compatibility
        output_spec = torch.stft(
            output_2d, 
            n_fft=window_size, 
            hop_length=hop_length, 
            window=window,
            return_complex=True
        )
        
        target_spec = torch.stft(
            target_2d, 
            n_fft=window_size, 
            hop_length=hop_length, 
            window=window,
            return_complex=True
        )
        
        # Convert complex tensor to magnitude for loss computation
        output_mag = torch.abs(output_spec)
        target_mag = torch.abs(target_spec)
        
        # Compute MSE loss on magnitudes
        return nn.functional.mse_loss(output_mag, target_mag)
    
    except RuntimeError as e:
        print(f"STFT error: {e}")
        print(f"Output shape: {output_2d.shape}, Target shape: {target_2d.shape}")
        # Fallback to L1 loss in time domain if STFT fails
        return nn.functional.l1_loss(output_2d, target_2d)

def train_separation_models(
    data_dir: str,
    output_dir: str,
    classifier_dir: str,
    batch_size: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    sample_rate: int = 44100,
    segment_duration: float = 5.0,
    device: torch.device = None
):
    """
    Train source separation models for each instrument class.
    
    Args:
        data_dir: Directory containing the raw data
        output_dir: Directory to save trained models
        classifier_dir: Directory containing trained classifier models
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        sample_rate: Audio sample rate
        segment_duration: Duration in seconds for each audio segment
        device: Device to train on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Loading data from: {data_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset with stem loading enabled
    dataset = InstrumentDataset(
        root_dir=data_dir,
        transform=None,
        sample_rate=sample_rate,
        n_mels=128,
        segment_duration=segment_duration,
        load_stems=True
    )
    
    # Get number of classes and class names
    num_classes = len(dataset.class_to_idx)
    class_names = list(dataset.class_to_idx.keys())
    print(f"Found {num_classes} instrument classes: {class_names}")
    
    # Split dataset into train and validation sets
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Find classifier models
    classifier_paths = []
    
    # Try ensemble directory first (most likely location)
    ensemble_dir = os.path.join(classifier_dir, 'ensemble')
    if os.path.exists(ensemble_dir) and os.path.isdir(ensemble_dir):
        # Look for model files in the ensemble directory
        model_files = [os.path.join(ensemble_dir, f) for f in os.listdir(ensemble_dir) 
                      if f.endswith('.pth')]
        if model_files:
            classifier_paths = model_files
            print(f"Found models in ensemble directory: {classifier_paths}")
    
    # If no models found in ensemble, try the standard locations
    if not classifier_paths:
        # Try ensemble models with standard naming
        ensemble_models = [os.path.join(classifier_dir, f"ensemble_model_{i+1}.pth") for i in range(5)]
        existing_ensemble_models = [path for path in ensemble_models if os.path.exists(path)]
        
        if existing_ensemble_models:
            classifier_paths = existing_ensemble_models
        else:
            # Try fold models
            fold_models = [os.path.join(classifier_dir, f"best_model_fold_{i+1}.pth") for i in range(5)]
            existing_fold_models = [path for path in fold_models if os.path.exists(path)]
            if existing_fold_models:
                classifier_paths = existing_fold_models
    
    if not classifier_paths:
        raise ValueError("No trained classifier models found. Please train the classifier first.")
    
    print(f"Found {len(classifier_paths)} classifier model(s): {classifier_paths}")
    
    # Train a separate model for each instrument class
    for idx, instrument in enumerate(class_names):
        print(f"\n{'='*50}")
        print(f"Training separation model for {instrument} ({idx+1}/{num_classes})")
        print(f"{'='*50}")
        
        # Create model for this instrument
        model = DemucsModel(sources=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=True
        )
        
        # Track best model
        best_loss = float('inf')
        best_model_path = os.path.join(output_dir, f"{instrument.replace(' ', '_')}_separator.pth")
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, (mix_waveforms, stems, labels) in enumerate(train_pbar):
                # Filter tracks that have this instrument
                instrument_present = labels[:, idx].bool()
                if not torch.any(instrument_present):
                    continue
                
                # Extract only tracks that have this instrument
                mix = mix_waveforms[instrument_present].to(device)
                
                # Get target and ensure it has the correct dimensions
                target_raw = stems[instrument_present][:, idx:idx+1].to(device)
                
                # Print size info for debugging on first batch
                if epoch == 0 and batch_idx == 0:
                    print(f"Target raw shape: {target_raw.shape}, size: {target_raw.numel()}")
                
                # Pre-process target to handle dimension issues safely
                if len(target_raw.shape) > 3:
                    # For 4D tensors, properly reshape based on actual dimensions and size
                    b = target_raw.size(0)  # Batch size
                    c = target_raw.size(1)  # Channels (1)
                    
                    # Calculate the time dimension by dividing total elements by batch and channels
                    total_elements = target_raw.numel()
                    t = total_elements // (b * c)
                    
                    # Reshape to 3D [batch, channels, time]
                    target = target_raw.reshape(b, c, t)
                else:
                    target = target_raw
                
                # Forward pass
                optimizer.zero_grad()
                output = model(mix)
                
                # Ensure output and target have the same shape
                output, target = ensure_matching_shapes(output, target)
                
                # Calculate L1 loss (time-domain)
                l1_loss = nn.functional.l1_loss(output, target)
                
                # Calculate spectral loss (frequency-domain)
                spec_loss = compute_spectral_loss(output, target, device)
                
                # Combined loss (0.8 * L1 + 0.2 * Spectral)
                loss = 0.8 * l1_loss + 0.2 * spec_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'avg_loss': f"{train_loss / (batch_idx + 1):.6f}"
                })
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            with torch.no_grad():
                for batch_idx, (mix_waveforms, stems, labels) in enumerate(val_pbar):
                    # Filter tracks that have this instrument
                    instrument_present = labels[:, idx].bool()
                    if not torch.any(instrument_present):
                        continue
                    
                    # Extract only tracks that have this instrument
                    mix = mix_waveforms[instrument_present].to(device)
                    
                    # Get target and ensure it has the correct dimensions
                    target_raw = stems[instrument_present][:, idx:idx+1].to(device)
                    
                    # Print size info for debugging on first batch
                    if epoch == 0 and batch_idx == 0:
                        print(f"Target raw shape: {target_raw.shape}, size: {target_raw.numel()}")
                    
                    # Pre-process target to handle dimension issues safely
                    if len(target_raw.shape) > 3:
                        # For 4D tensors, properly reshape based on actual dimensions and size
                        b = target_raw.size(0)  # Batch size
                        c = target_raw.size(1)  # Channels (1)
                        
                        # Calculate the time dimension by dividing total elements by batch and channels
                        total_elements = target_raw.numel()
                        t = total_elements // (b * c)
                        
                        # Reshape to 3D [batch, channels, time]
                        target = target_raw.reshape(b, c, t)
                    else:
                        target = target_raw
                    
                    # Forward pass
                    output = model(mix)
                    
                    # Ensure output and target have the same shape
                    output, target = ensure_matching_shapes(output, target)
                    
                    # Calculate L1 loss
                    l1_loss = nn.functional.l1_loss(output, target)
                    
                    # Calculate spectral loss
                    spec_loss = compute_spectral_loss(output, target, device)
                    
                    # Combined loss
                    loss = 0.8 * l1_loss + 0.2 * spec_loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f"{loss.item():.6f}",
                        'avg_loss': f"{val_loss / (batch_idx + 1):.6f}"
                    })
            
            # Calculate average losses
            avg_train_loss = train_loss / max(train_batches, 1)
            avg_val_loss = val_loss / max(val_batches, 1)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")
        
        print(f"Training complete for {instrument}")
    
    print("\nTraining complete for all instruments.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train instrument separation models')
    parser.add_argument('--data_dir', type=str, help='Directory containing the raw data')
    parser.add_argument('--output_dir', type=str, default='./separation_models', 
                        help='Directory to save trained models')
    parser.add_argument('--classifier_dir', type=str, default='./', 
                        help='Directory containing trained classifier models')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sample_rate', type=int, default=44100, help='Audio sample rate')
    parser.add_argument('--segment_duration', type=float, default=5.0, 
                        help='Duration in seconds for each audio segment')
    args = parser.parse_args()
    
    # Get project root if data_dir not specified
    if args.data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../..'))
        args.data_dir = os.path.join(project_root, 'data', 'raw')
    
    # Train separation models
    train_separation_models(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        classifier_dir=args.classifier_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        sample_rate=args.sample_rate,
        segment_duration=args.segment_duration
    )

if __name__ == '__main__':
    main() 