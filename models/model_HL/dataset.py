import os
import torch
from torch.utils.data import Dataset
import torchaudio
import yaml
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class InstrumentDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, sample_rate: int = 44100, n_mels: int = 128, 
                 segment_duration: float = 5.0, load_stems: bool = False):
        """
        Dataset for instrument classification from audio files.
        
        Args:
            root_dir (str): Root directory containing the raw data
            transform: Optional transform to be applied on the audio
            sample_rate (int): Target sample rate for audio
            n_mels (int): Number of mel bands for spectrogram
            segment_duration (float): Duration in seconds for each audio segment
            load_stems (bool): Whether to load individual stems for source separation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.segment_duration = segment_duration
        self.target_length = int(segment_duration * sample_rate)
        self.load_stems = load_stems
        
        # Calculate expected time dimension of mel spectrogram
        hop_length = 512
        self.target_time_steps = int((self.target_length / hop_length) + 1)
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Load all track directories
        self.track_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Create instrument class mapping
        self.class_to_idx = self._create_class_mapping()
        
    def _create_class_mapping(self) -> Dict[str, int]:
        """Create a mapping from instrument class names to indices."""
        classes = set()
        for track_dir in self.track_dirs:
            metadata_path = os.path.join(self.root_dir, track_dir, 'metadata.yaml')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f)
                    for stem in metadata['stems'].values():
                        if 'inst_class' in stem:
                            classes.add(stem['inst_class'])
        
        return {cls: idx for idx, cls in enumerate(sorted(classes))}
    
    def _load_and_process_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio and ensure it has a fixed length."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ensure the waveform has consistent length
        current_length = waveform.shape[1]
        
        if current_length >= self.target_length:
            # If the audio is long enough, take a random segment
            if current_length > self.target_length:
                max_start_idx = current_length - self.target_length
                start_idx = torch.randint(0, max_start_idx, (1,)).item()
                waveform = waveform[:, start_idx:start_idx + self.target_length]
        else:
            # If the audio is too short, pad with zeros
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform
    
    def _get_labels(self, metadata_path: str) -> torch.Tensor:
        """Get multi-hot encoded labels from metadata."""
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Initialize label tensor with zeros
        labels = torch.zeros(len(self.class_to_idx))
        
        # Set 1 for each instrument class present
        for stem in metadata['stems'].values():
            if 'inst_class' in stem:
                class_idx = self.class_to_idx[stem['inst_class']]
                labels[class_idx] = 1
        
        return labels
    
    def _get_stem_map(self, metadata_path: str) -> Dict[str, List[str]]:
        """
        Maps instrument classes to stem files.
        
        Returns:
            Dictionary mapping instrument class to list of stem files
        """
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        stem_map = {}
        for stem_id, stem_data in metadata['stems'].items():
            if 'inst_class' in stem_data:
                inst_class = stem_data['inst_class']
                if inst_class not in stem_map:
                    stem_map[inst_class] = []
                stem_map[inst_class].append(stem_id)
        
        return stem_map
    
    def _load_stems(self, track_dir: str, metadata_path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load individual instrument stems for a track.
        
        Args:
            track_dir: Directory containing the track
            metadata_path: Path to the metadata file
            
        Returns:
            Tuple containing:
                - Dictionary mapping instrument class to stem audio
                - Tensor of shape [num_classes, 1, time] with separated audio
        """
        # Get mapping of instrument classes to stem files
        stem_map = self._get_stem_map(metadata_path)
        
        # Get random segment offset for consistent segment extraction
        mix_path = os.path.join(self.root_dir, track_dir, 'mix.wav')
        waveform, sr = torchaudio.load(mix_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        current_length = waveform.shape[1]
        start_idx = 0
        if current_length > self.target_length:
            max_start_idx = current_length - self.target_length
            start_idx = torch.randint(0, max_start_idx, (1,)).item()
        
        # Initialize tensor for all stems
        all_stems = torch.zeros(len(self.class_to_idx), 1, self.target_length)
        
        # Load each stem
        stems_dir = os.path.join(self.root_dir, track_dir, 'stems')
        if os.path.exists(stems_dir):
            for inst_class, stem_ids in stem_map.items():
                class_idx = self.class_to_idx[inst_class]
                
                # Combine multiple stems of the same instrument class
                combined_stem = torch.zeros(1, self.target_length)
                
                for stem_id in stem_ids:
                    stem_path = os.path.join(stems_dir, f"{stem_id}.wav")
                    if os.path.exists(stem_path):
                        stem_waveform, stem_sr = torchaudio.load(stem_path)
                        
                        # Resample if necessary
                        if stem_sr != self.sample_rate:
                            stem_resampler = torchaudio.transforms.Resample(stem_sr, self.sample_rate)
                            stem_waveform = stem_resampler(stem_waveform)
                        
                        # Convert to mono if stereo
                        if stem_waveform.shape[0] > 1:
                            stem_waveform = torch.mean(stem_waveform, dim=0, keepdim=True)
                        
                        # Extract the same segment as the mix
                        if stem_waveform.shape[1] > start_idx + self.target_length:
                            stem_segment = stem_waveform[:, start_idx:start_idx + self.target_length]
                        else:
                            # Handle shorter stems by padding
                            usable_length = min(stem_waveform.shape[1] - start_idx, self.target_length)
                            if usable_length > 0:
                                stem_segment = torch.zeros(1, self.target_length)
                                stem_segment[:, :usable_length] = stem_waveform[:, start_idx:start_idx + usable_length]
                            else:
                                # Skip if stem is too short
                                continue
                        
                        # Add to combined stem
                        combined_stem += stem_segment
                
                # Normalize the combined stem
                if torch.max(torch.abs(combined_stem)) > 0:
                    combined_stem = combined_stem / torch.max(torch.abs(combined_stem))
                
                # Store in all_stems tensor
                all_stems[class_idx] = combined_stem
        
        return stem_map, all_stems
    
    def __len__(self) -> int:
        return len(self.track_dirs)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        track_dir = self.track_dirs[idx]
        
        # Load mix.wav
        audio_path = os.path.join(self.root_dir, track_dir, 'mix.wav')
        waveform = self._load_and_process_audio(audio_path)
        
        # Get labels from metadata
        metadata_path = os.path.join(self.root_dir, track_dir, 'metadata.yaml')
        labels = self._get_labels(metadata_path)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)  # Log-mel spectrogram
        
        # Ensure consistent time dimension for the spectrogram
        current_time_steps = mel_spec.shape[2]
        if current_time_steps < self.target_time_steps:
            # Pad if needed
            padding = self.target_time_steps - current_time_steps
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
        elif current_time_steps > self.target_time_steps:
            # Truncate if needed
            mel_spec = mel_spec[:, :, :self.target_time_steps]
        
        # Apply any additional transforms
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        # For classification only, return mel spectrogram and labels
        if not self.load_stems:
            return mel_spec, labels
        
        # For source separation, also load stems
        _, stem_waveforms = self._load_stems(track_dir, metadata_path)
        
        return waveform, stem_waveforms, labels 