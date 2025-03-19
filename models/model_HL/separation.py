import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class DemucsEncoder(nn.Module):
    """Encoder part of the Demucs model for source separation."""
    def __init__(self, in_channels, channels, kernel_size, stride, depth):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            out_channels = channels * (2 ** i)
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels if i == 0 else channels * (2 ** (i - 1)),
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels)
                )
            )
        
    def forward(self, x):
        # Store skip connections
        skips = []
        print(f"Encoder input shape: {x.shape}")
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"Encoder layer {i} output shape: {x.shape}")
            skips.append(x)
        return x, skips

class DemucsDecoder(nn.Module):
    """Decoder part of the Demucs model for source separation."""
    def __init__(self, channels, kernel_size, stride, depth, sources=1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Calculate initial channels for decoder
        initial_channels = channels * (2 ** (depth - 1))
        
        # Store expected output sizes for each layer to track issues
        self.expected_output_sizes = []
        
        for i in range(depth):
            in_channels = initial_channels // (2 ** i)
            if i == 0:
                out_channels = in_channels // 2
            else:
                out_channels = in_channels // 2
            
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels * 2 if i > 0 else in_channels,  # Double for skip
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        output_padding=stride - 1
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels)
                )
            )
        
        # Calculate the actual channel count of the last decoder layer's output
        final_decoder_channels = initial_channels // (2 ** (depth - 1)) // 2
        # Final layer to generate source - match the actual channel count
        self.final = nn.Conv1d(final_decoder_channels, sources, kernel_size=3, padding=1)
    
    def _match_size(self, x, target):
        """
        Adjust the size of x to match the target size in the time dimension (dim 2).
        Either by padding with zeros or cropping.
        """
        size_x = x.shape[2]
        size_target = target.shape[2]
        
        print(f"Matching sizes: x={x.shape}, target={target.shape}")
        
        if size_x > size_target:
            # Crop the input tensor
            diff = size_x - size_target
            x = x[:, :, :size_target]
            print(f"Cropped x to {x.shape}, removed {diff} samples")
        elif size_x < size_target:
            # Pad the input tensor
            diff = size_target - size_x
            x = F.pad(x, (0, diff))
            print(f"Padded x to {x.shape}, added {diff} samples")
        
        return x
    
    def _calculate_expected_output_size(self, input_size, stride, kernel_size, padding, output_padding):
        """Calculate the expected output size for ConvTranspose1d operation"""
        return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    
    def forward(self, x, skips):
        # Go through layers with skip connections
        print(f"Decoder input shape: {x.shape}")
        
        # Store layer inputs and outputs for debugging
        layer_inputs = []
        layer_outputs = []
        
        for i, layer in enumerate(self.layers):
            layer_inputs.append(x.clone())
            
            if i > 0:  # Skip for first layer as it doesn't have a corresponding encoder layer
                # Concatenate with skip connection
                skip = skips[-(i+1)]
                print(f"Skip connection {i} shape: {skip.shape}")
                
                # Ensure x and skip have the same size in the time dimension
                if x.shape[2] != skip.shape[2]:
                    # If sizes don't match, adjust x to match skip size
                    print(f"Size mismatch before adjustment: x={x.shape}, skip={skip.shape}")
                    x = self._match_size(x, skip)
                    
                    # If we still have a mismatch (rare edge case), adjust skip
                    if x.shape[2] != skip.shape[2]:
                        print(f"Still mismatched after x adjustment: x={x.shape}, skip={skip.shape}")
                        skip = self._match_size(skip, x)
                    print(f"After adjustment: x={x.shape}, skip={skip.shape}")
                
                x = torch.cat([x, skip], dim=1)
                print(f"After concatenation: {x.shape}")
            
            # Calculate expected output size for convtranspose
            if isinstance(layer[0], nn.ConvTranspose1d):
                conv = layer[0]
                expected_output_size = self._calculate_expected_output_size(
                    x.shape[2], 
                    conv.stride[0], 
                    conv.kernel_size[0],
                    conv.padding[0],
                    conv.output_padding[0]
                )
                print(f"Layer {i} - Expected output size: {expected_output_size}")
            
            # Process through layer
            x = layer(x)
            layer_outputs.append(x.clone())
            
            # Check actual output size
            print(f"Layer {i} - Actual output shape: {x.shape}")
            
            # Compare with expected size
            if isinstance(layer[0], nn.ConvTranspose1d) and x.shape[2] != expected_output_size:
                print(f"WARNING: Actual size {x.shape[2]} differs from expected {expected_output_size}")
                print(f"Difference: {x.shape[2] - expected_output_size} samples")
        
        print(f"Before final layer: {x.shape}")
        output = self.final(x)
        print(f"Final decoder output shape: {output.shape}")
        
        return output

class DemucsModel(nn.Module):
    """Simplified Demucs model for source separation with exact dimension matching."""
    def __init__(self, 
                 sources: int = 1,
                 channels: int = 64, 
                 kernel_size: int = 8, 
                 stride: int = 4, 
                 depth: int = 4,  # Reduced depth for simpler architecture
                 debug: bool = False):
        """
        Initialize Demucs model for source separation with exact dimension matching.
        
        Args:
            sources: Number of sources to extract (one per instrument)
            channels: Initial number of channels
            kernel_size: Kernel size for convolutions
            stride: Stride for convolutions
            depth: Number of encoder/decoder blocks
            debug: Whether to print debug information
        """
        super().__init__()
        # Use simpler architecture with guaranteed shape preservation
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        # Create encoder layers
        in_channels = 1
        for i in range(depth):
            out_channels = channels * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels)
                )
            )
            in_channels = out_channels
        
        # Create decoder layers (in reverse order)
        for i in range(depth):
            # Last layer outputs to source channels
            if i == depth - 1:
                out_channels = sources
            else:
                out_channels = channels * (2 ** (depth - i - 2))
            
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        channels * (2 ** (depth - i - 1)),
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        output_padding=stride - 1
                    ),
                    nn.ReLU() if i < depth - 1 else nn.Identity(),  # No activation on final layer
                    nn.BatchNorm1d(out_channels) if i < depth - 1 else nn.Identity()  # No BN on final layer
                )
            )
        
        self.debug = debug
        
    def forward(self, mix_waveform):
        """
        Forward pass of the model with guaranteed shape preservation.
        
        Args:
            mix_waveform: Mixed waveform input [batch, 1, time]
            
        Returns:
            Separated source waveforms [batch, sources, time] with exactly the same time dimension
        """
        # Store original input for shape comparison
        original_input = mix_waveform
        input_length = original_input.shape[2]
        
        if self.debug:
            print(f"Input waveform shape: {mix_waveform.shape}")
        
        # Store skip connections for symmetric U-Net style architecture
        skips = []
        
        # Encoder forward pass with skip connections
        x = mix_waveform
        for i, layer in enumerate(self.encoder):
            # Store input shape before encoding
            if self.debug:
                print(f"Encoder layer {i} input shape: {x.shape}")
            
            # Apply encoder layer
            x = layer(x)
            
            # Store skip connection
            skips.append(x)
            
            if self.debug:
                print(f"Encoder layer {i} output shape: {x.shape}")
        
        # Decoder forward pass with skip connections
        for i, layer in enumerate(self.decoder):
            if self.debug:
                print(f"Decoder layer {i} input shape: {x.shape}")
            
            # Apply decoder layer
            x = layer(x)
            
            if self.debug:
                print(f"Decoder layer {i} output shape: {x.shape}")
        
        # Ensure output has exactly the same length as input
        if x.shape[2] != input_length:
            if self.debug:
                print(f"Output size mismatch: got {x.shape[2]}, expected {input_length}")
                print(f"Difference: {x.shape[2] - input_length} samples")
            
            # Exact shape matching by cropping or padding
            if x.shape[2] > input_length:
                x = x[:, :, :input_length]
                if self.debug:
                    print(f"Cropped output to match input size: {x.shape}")
            else:
                padding = input_length - x.shape[2]
                x = F.pad(x, (0, padding))
                if self.debug:
                    print(f"Padded output to match input size: {x.shape}")
        
        if self.debug:
            print(f"Final output shape: {x.shape}")
            print(f"Original input shape: {original_input.shape}")
            assert x.shape[2] == original_input.shape[2], "Output and input shapes don't match!"
        
        return x

class InstrumentSeparator:
    """
    Manages the separation of instruments from a mix using trained models.
    """
    def __init__(self, 
                 classifier_paths: List[str],
                 instrument_classes: List[str],
                 sample_rate: int = 44100,
                 device: Optional[torch.device] = None):
        """
        Initialize the instrument separator.
        
        Args:
            classifier_paths: Paths to the trained classifier models
            instrument_classes: List of instrument class names
            sample_rate: Audio sample rate
            device: Device to run the models on
        """
        # Store original classes for model loading
        self.original_classes = instrument_classes.copy()
        
        # Normalize and combine similar instrument classes
        self.instrument_classes = self._normalize_instrument_classes(instrument_classes)
        self.sample_rate = sample_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create class mapping from original to normalized classes
        self.class_mapping = self._create_class_mapping(instrument_classes)
        
        # Load classifier models
        from model import create_model
        self.classifier_models = []
        for path in classifier_paths:
            model = create_model(num_classes=len(self.original_classes))
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.classifier_models.append(model)
        
        # Create separation models (one per instrument class)
        self.separation_models = {}
        
        # Will be populated in the train_separation_models method
    
    def _normalize_instrument_classes(self, classes):
        """
        Normalize instrument class names and combine similar classes.
        
        Args:
            classes: List of instrument class names
            
        Returns:
            Normalized list of class names with duplicates combined
        """
        # Create a mapping for classes to combine
        class_mapping = {}
        normalized_classes = []
        
        # Handle specific merges
        sound_effects_found = False
        strings_found = False
        
        for cls in classes:
            # Normalize case
            norm_cls = cls.strip()
            
            # Handle Sound Effects variants
            if norm_cls.lower() == "sound effects":
                if not sound_effects_found:
                    normalized_classes.append("Sound Effects")
                    sound_effects_found = True
                class_mapping[cls] = "Sound Effects"
            
            # Handle Strings variants
            elif norm_cls.lower() == "strings" or norm_cls.lower() == "strings (continued)":
                if not strings_found:
                    normalized_classes.append("Strings")
                    strings_found = True
                class_mapping[cls] = "Strings"
            
            # Keep other classes as is
            elif norm_cls not in class_mapping:
                normalized_classes.append(norm_cls)
                class_mapping[cls] = norm_cls
        
        print(f"Original classes: {len(classes)}, Normalized classes: {len(normalized_classes)}")
        print(f"Normalized instrument classes: {normalized_classes}")
        
        return normalized_classes
    
    def _create_class_mapping(self, original_classes):
        """Create mapping from original class indices to normalized class indices"""
        mapping = {}
        
        for i, cls in enumerate(original_classes):
            norm_cls = cls.strip()
            
            # Map Sound Effects variants
            if norm_cls.lower() == "sound effects":
                target_idx = self.instrument_classes.index("Sound Effects")
                mapping[i] = target_idx
            
            # Map Strings variants
            elif norm_cls.lower() == "strings" or norm_cls.lower() == "strings (continued)":
                target_idx = self.instrument_classes.index("Strings")
                mapping[i] = target_idx
            
            # Map other classes
            else:
                try:
                    target_idx = self.instrument_classes.index(norm_cls)
                    mapping[i] = target_idx
                except ValueError:
                    # If not in normalized classes, map to itself
                    mapping[i] = i
        
        return mapping
    
    def classify_instruments(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Predict which instruments are present in the mix.
        
        Args:
            waveform: Audio waveform [1, time]
            
        Returns:
            Binary tensor indicating instrument presence [num_classes]
        """
        # Process audio to create mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # Convert to mel spectrogram
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)  # Log-mel spectrogram
        
        # Add batch dimension for model input
        mel_spec = mel_spec.unsqueeze(0)  # [1, freq, time]
        
        # Get predictions from all models
        all_probs = []
        with torch.no_grad():
            for model in self.classifier_models:
                outputs = model(mel_spec.to(self.device))
                probs = torch.sigmoid(outputs)
                all_probs.append(probs)
        
        # Average predictions from ensemble
        avg_ensemble_probs = torch.stack(all_probs).mean(dim=0).squeeze(0)
        
        # Create tensor for normalized classes
        normalized_probs = torch.zeros(len(self.instrument_classes), device=self.device)
        
        # Map and combine probabilities for merged classes
        for orig_idx, norm_idx in self.class_mapping.items():
            if orig_idx < avg_ensemble_probs.size(0):
                # For merged classes, use max probability
                normalized_probs[norm_idx] = max(normalized_probs[norm_idx], avg_ensemble_probs[orig_idx])
        
        # Convert to binary predictions
        binary_preds = (normalized_probs > 0.5).float()
        
        return binary_preds
    
    def train_separation_models(self, 
                               train_dataset,
                               num_epochs: int = 100,
                               batch_size: int = 16,
                               lr: float = 1e-3,
                               save_dir: str = './separation_models'):
        """
        Train separation models for each instrument class.
        
        Args:
            train_dataset: Dataset containing mix.wav and individual stems
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            save_dir: Directory to save trained models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a DataLoader for training
        from torch.utils.data import DataLoader
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Helper function to match tensor sizes using truncation approach with advanced diagnostics
        def match_tensor_sizes(output, target, debug=False):
            """Match the temporal dimension of output to target using truncation with diagnostics."""
            # Remove extra dimensions from target if present
            if len(target.shape) > 3:
                original_target_shape = target.shape
                target = target.squeeze(-1)
                if debug:
                    print(f"Squeezed target from {original_target_shape} to {target.shape}")
            
            # Make copy of tensors for analysis
            orig_output = output.clone()
            orig_target = target.clone()
            
            if output.shape != target.shape:
                if debug:
                    print(f"Before matching: output={output.shape}, target={target.shape}")
                    # Analyze dimensions with differences
                    shape_diffs = [o - t for o, t in zip(output.shape, target.shape)]
                    print(f"Dimension differences (output - target): {shape_diffs}")
                
                # Get the time dimension (last dimension)
                output_time = output.shape[-1]
                target_time = target.shape[-1]
                
                # Store the size differences
                time_diff = output_time - target_time
                
                # Find minimum length and truncate both tensors
                min_length = min(output_time, target_time)
                output = output[..., :min_length]
                target = target[..., :min_length]
                
                if debug:
                    print(f"After matching: output={output.shape}, target={target.shape}")
                    print(f"Truncated time dimension difference: {time_diff} samples")
                    if abs(time_diff) == 3:
                        print(f"FOUND THE ISSUE: 3-sample difference detected in time dimension")
                        # Detailed analysis of the tensor sizes
                        print(f"Original sizes: output={orig_output.shape}, target={orig_target.shape}")
                        # Track the size through the model
                        print(f"This is likely caused by ConvTranspose1d layers with asymmetric padding")
            
            # Final assertion to guarantee shapes match
            try:
                assert output.shape == target.shape, f"Shapes still don't match: output={output.shape}, target={target.shape}"
            except AssertionError as e:
                print(f"ERROR: {e}")
                # Try more aggressive truncation as a last resort
                common_shape = [min(o, t) for o, t in zip(output.shape, target.shape)]
                print(f"Applying emergency shape correction to common shape: {common_shape}")
                # Apply slicing to all dimensions
                output_slices = tuple(slice(0, dim) for dim in common_shape)
                target_slices = tuple(slice(0, dim) for dim in common_shape)
                output = output[output_slices]
                target = target[target_slices]
                print(f"After emergency correction: output={output.shape}, target={target.shape}")
            
            return output, target
        
        # For tracking consistent tensor size issues across batches
        size_difference_counter = {}
        
        # Train a separate model for each normalized instrument class
        for norm_idx, instrument in enumerate(self.instrument_classes):
            print(f"Training separation model for {instrument}...")
            
            # Create a model for this instrument
            model = DemucsModel(sources=1).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5, verbose=True
            )
            
            best_loss = float('inf')
            
            for epoch in range(num_epochs):
                # Training loop
                model.train()
                epoch_loss = 0
                batch_count = 0
                
                for batch_idx, (mix_spec, stems, labels) in enumerate(loader):
                    # Enable debug mode for first batch of first epoch or if issues detected
                    debug_mode = (epoch == 0 and batch_idx == 0)
                    if debug_mode:
                        print(f"\n==== DEBUG MODE ENABLED FOR FIRST BATCH ====")
                        print(f"Input mix_spec shape: {mix_spec.shape}")
                        print(f"Stems shape: {stems.shape}")
                        print(f"Labels shape: {labels.shape}")
                        
                        # Set model to debug mode
                        model.debug = True
                    else:
                        model.debug = False
                    
                    # Find original indices that map to this normalized instrument
                    orig_indices = [i for i, idx in self.class_mapping.items() if idx == norm_idx]
                    
                    # Get tracks where any of the mapped original instruments are present
                    instrument_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
                    for orig_idx in orig_indices:
                        if orig_idx < labels.shape[1]:
                            instrument_mask = instrument_mask | labels[:, orig_idx].bool()
                    
                    if not torch.any(instrument_mask):
                        continue
                    
                    # Extract only tracks that have this instrument
                    mix = mix_spec[instrument_mask].to(self.device)
                    
                    # For target, combine stems from all mapped original instruments
                    # Initialize target with zeros
                    target = torch.zeros((instrument_mask.sum(), 1, stems.shape[2], stems.shape[3]), device=self.device)
                    for orig_idx in orig_indices:
                        if orig_idx < stems.shape[1]:
                            # Add the stem to target
                            stem_subset = stems[instrument_mask][:, orig_idx:orig_idx+1].to(self.device)
                            target = target + stem_subset
                    
                    if debug_mode:
                        print(f"Filtered mix shape: {mix.shape}")
                        print(f"Target shape: {target.shape}")
                    
                    # Forward pass - with exception handling for shape issues
                    optimizer.zero_grad()
                    try:
                        output = model(mix)
                        
                        if debug_mode:
                            print(f"Model output shape: {output.shape}")
                            print(f"Target tensor shape: {target.shape}")
                        
                        # Record size difference for analysis
                        if output.shape[-1] != target.squeeze(-1).shape[-1]:
                            diff = output.shape[-1] - target.squeeze(-1).shape[-1]
                            size_diff_key = f"{output.shape[-1]}-{target.squeeze(-1).shape[-1]}={diff}"
                            size_difference_counter[size_diff_key] = size_difference_counter.get(size_diff_key, 0) + 1
                            
                            # Print only for significant counts or the first batch
                            if size_difference_counter[size_diff_key] == 1 or debug_mode:
                                print(f"Size difference detected: {size_diff_key}, count: {size_difference_counter[size_diff_key]}")
                        
                        # Ensure output and target have the same shape
                        output, target = match_tensor_sizes(output, target, debug=debug_mode)
                        
                        if debug_mode:
                            print(f"Final matched shapes - output: {output.shape}, target: {target.shape}")
                        
                        # Calculate loss (L1 loss works well for audio)
                        try:
                            loss = F.l1_loss(output, target)
                        except RuntimeError as e:
                            print(f"Error computing loss: {e}")
                            print(f"Output shape: {output.shape}, Target shape: {target.shape}")
                            raise
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                        
                    except RuntimeError as e:
                        print(f"Runtime error during forward/backward pass: {e}")
                        print(f"Input mix shape: {mix.shape}")
                        
                        # Skip this batch but continue training
                        print(f"Skipping batch {batch_idx} due to error")
                        continue
                    
                    if debug_mode:
                        print(f"==== DEBUG MODE COMPLETED ====\n")
                
                # Print size difference statistics at the end of each epoch
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    print("\nTensor size difference statistics:")
                    for diff_key, count in sorted(size_difference_counter.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {diff_key}: {count} occurrences")
                    print()
                
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
                    
                    # Update scheduler
                    scheduler.step(avg_loss)
                    
                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(model.state_dict(), 
                                  os.path.join(save_dir, f"{instrument.replace(' ', '_')}_separator.pth"))
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}: No batches with {instrument} found")
            
            # Print final size difference summary
            print("\nFinal tensor size difference summary:")
            for diff_key, count in sorted(size_difference_counter.items(), key=lambda x: x[1], reverse=True):
                print(f"  {diff_key}: {count} occurrences")
            print()
            
            # Load the best model if training occurred
            model_path = os.path.join(save_dir, f"{instrument.replace(' ', '_')}_separator.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                self.separation_models[instrument] = model
    
    def load_separation_models(self, model_dir: str):
        """
        Load pretrained separation models.
        
        Args:
            model_dir: Directory containing the saved models
        """
        for instrument in self.instrument_classes:
            model_path = os.path.join(model_dir, f"{instrument.replace(' ', '_')}_separator.pth")
            if os.path.exists(model_path):
                model = DemucsModel(sources=1).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                self.separation_models[instrument] = model
                print(f"Loaded separation model for {instrument}")
            else:
                print(f"Warning: No separation model found for {instrument}")
    
    def separate_instruments(self, 
                            mix_waveform: torch.Tensor, 
                            instrument_predictions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Separate individual instruments from a mix.
        
        Args:
            mix_waveform: Mixed waveform [1, time]
            instrument_predictions: Optional binary tensor of instrument predictions
                                   If None, will use the classifier to predict
            
        Returns:
            Dictionary mapping instrument names to separated waveforms
        """
        if instrument_predictions is None:
            instrument_predictions = self.classify_instruments(mix_waveform)
        
        # Create dictionary to store separated waveforms
        separated_waveforms = {}
        
        # Original input length
        orig_length = mix_waveform.shape[-1]
        
        # Process each predicted instrument
        with torch.no_grad():
            for idx, instrument in enumerate(self.instrument_classes):
                if instrument_predictions[idx] > 0.5 and instrument in self.separation_models:
                    # Get the separation model for this instrument
                    model = self.separation_models[instrument]
                    
                    # Separate this instrument
                    waveform = mix_waveform.to(self.device)
                    separated = model(waveform.unsqueeze(0)).squeeze(0)
                    
                    # Match length using truncation to ensure consistent behavior
                    sep_length = separated.shape[-1]
                    min_length = min(sep_length, orig_length)
                    
                    # Truncate both to the minimum length
                    separated = separated[..., :min_length]
                    
                    # Store the separated waveform
                    separated_waveforms[instrument] = separated.cpu()
        
        return separated_waveforms
    
    def process_audio_file(self, mix_path: str, output_dir: str) -> Dict[str, str]:
        """
        Process an audio file, separate instruments, and save the results.
        
        Args:
            mix_path: Path to the mixed audio file
            output_dir: Directory to save separated audio files
            
        Returns:
            Dictionary mapping instrument names to output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio file
        waveform, sr = torchaudio.load(mix_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Classify instruments using normalized classes
        instrument_predictions = self.classify_instruments(waveform)
        predicted_instruments = [self.instrument_classes[i] for i in range(len(self.instrument_classes)) 
                                if instrument_predictions[i] > 0.5]
        
        print(f"Detected instruments: {', '.join(predicted_instruments)}")
        
        # Separate instruments
        separated_waveforms = self.separate_instruments(waveform, instrument_predictions)
        
        # Save separated waveforms
        output_paths = {}
        for instrument, separated in separated_waveforms.items():
            output_path = os.path.join(output_dir, f"{instrument.replace(' ', '_')}.wav")
            torchaudio.save(output_path, separated, self.sample_rate)
            output_paths[instrument] = output_path
            print(f"Saved separated {instrument} to {output_path}")
        
        return output_paths 