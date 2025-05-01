import os
import glob
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from torch.utils.data import Dataset, DataLoader
import logging
import re
from transforms import AudioTransforms

class SpeakerDataset(Dataset):
    """Dataset for speaker classification from audio files."""
    
    def __init__(self, 
                 root_dir: str, 
                 transform: Optional[Callable] = None,
                 split: str = 'train',
                 train_ratio: float = 0.9,
                 seed: int = 42):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory with all speaker folders
            transform: Optional transform to apply to audio files
            split: Split to use ('train' or 'val')
            train_ratio: Ratio of train samples to total
            seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.transform = transform or AudioTransforms(apply_augmentation=(split == 'train'))
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Get all speaker directories
        speaker_dirs = sorted(glob.glob(os.path.join(root_dir, "person*")))
        self.num_speakers = len(speaker_dirs)
        
        # Create mapping from speaker name to ID
        self.speaker_to_idx = {os.path.basename(path): idx for idx, path in enumerate(speaker_dirs)}
        self.idx_to_speaker = {idx: os.path.basename(path) for idx, path in enumerate(speaker_dirs)}
        
        # Get all audio files and labels
        self.audio_files = []
        self.labels = []
        self.file_ids = []
        
        # Regex pattern to extract file ID
        file_id_pattern = re.compile(r'chunk_(\d+)\.wav')
        
        for speaker_dir in speaker_dirs:
            speaker_name = os.path.basename(speaker_dir)
            speaker_files = sorted(glob.glob(os.path.join(speaker_dir, "chunk_*.wav")))
            
            for file_path in speaker_files:
                # Extract file ID from filename
                file_id_match = file_id_pattern.search(os.path.basename(file_path))
                if file_id_match:
                    file_id = f"chunk_{file_id_match.group(1)}.wav"
                    self.file_ids.append(file_id)
                    self.audio_files.append(file_path)
                    self.labels.append(self.speaker_to_idx[speaker_name])
        
        # Create deterministic train/val split
        indices = np.arange(len(self.audio_files))
        np.random.shuffle(indices)
        
        train_size = int(self.train_ratio * len(indices))
        
        if split == 'train':
            self.indices = indices[:train_size]
        else:  # val split
            self.indices = indices[train_size:]
        
        # Subset data based on split
        self.audio_files = [self.audio_files[i] for i in self.indices]
        self.labels = [self.labels[i] for i in self.indices]
        self.file_ids = [self.file_ids[i] for i in self.indices]
        
        logging.info(f"Created {split} dataset with {len(self.audio_files)} samples "
                    f"from {self.num_speakers} speakers")
    
    def __len__(self) -> int:
        """Return the number of audio files in the dataset."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with 'spectrogram', 'label', and 'file_id'
        """
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        file_id = self.file_ids[idx]
        
        # Apply transforms to get spectrogram
        spectrogram = self.transform(audio_path)
        
        # Ensure spectrogram has the right shape
        if spectrogram.dim() == 3:  # (1, n_mels, time)
            # Convert to (n_mels, time) for model input
            spectrogram = spectrogram.squeeze(0)
        
        # Ensure label is a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'spectrogram': spectrogram,
            'label': label,
            'file_id': file_id
        }

class TestDataset(Dataset):
    """Dataset for generating predictions on unseen test files."""
    
    def __init__(self, 
                 test_dir: str,
                 transform: Optional[Callable] = None,
                 known_speakers: Optional[Dict[str, int]] = None):
        """
        Initialize the test dataset.
        
        Args:
            test_dir: Directory with test audio files
            transform: Transform to apply to audio files
            known_speakers: Mapping from speaker names to indices
        """
        self.test_dir = test_dir
        self.transform = transform or AudioTransforms(apply_augmentation=False)
        self.known_speakers = known_speakers or {}
        
        # Get all test audio files
        self.audio_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
        self.file_ids = [os.path.basename(f) for f in self.audio_files]
        
        logging.info(f"Created test dataset with {len(self.audio_files)} samples")
    
    def __len__(self) -> int:
        """Return the number of audio files in the dataset."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with 'spectrogram' and 'file_id'
        """
        audio_path = self.audio_files[idx]
        file_id = self.file_ids[idx]
        
        # Apply transforms to get spectrogram
        spectrogram = self.transform(audio_path)
        
        # Ensure spectrogram has the right shape
        if spectrogram.dim() == 3:  # (1, n_mels, time)
            # Convert to (n_mels, time) for model input
            spectrogram = spectrogram.squeeze(0)
        
        return {
            'spectrogram': spectrogram,
            'file_id': file_id
        }

def create_dataloaders(root_dir: str, 
                       batch_size: int = 32,
                       num_workers: int = 4,
                       train_ratio: float = 0.9,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        root_dir: Root directory with speaker folders
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        train_ratio: Ratio of train samples to total
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create train dataset and dataloader
    train_dataset = SpeakerDataset(
        root_dir=root_dir,
        transform=AudioTransforms(apply_augmentation=True),
        split='train',
        train_ratio=train_ratio,
        seed=seed
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Create validation dataset and dataloader
    val_dataset = SpeakerDataset(
        root_dir=root_dir,
        transform=AudioTransforms(apply_augmentation=False),
        split='val',
        train_ratio=train_ratio,
        seed=seed
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader, train_dataset.speaker_to_idx

def create_test_dataloader(test_dir: str,
                          known_speakers: Dict[str, int],
                          batch_size: int = 32,
                          num_workers: int = 4) -> DataLoader:
    """
    Create test dataloader.
    
    Args:
        test_dir: Directory with test audio files
        known_speakers: Mapping from speaker names to indices
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        
    Returns:
        Test dataloader
    """
    test_dataset = TestDataset(
        test_dir=test_dir,
        transform=AudioTransforms(apply_augmentation=False),
        known_speakers=known_speakers
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return test_dataloader 