import os
import sys
import wave
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

def extract_features(audio_file: str) -> np.ndarray:
    """
    Extract basic audio features from a WAV file.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Feature vector
    """
    try:
        # Open the wave file
        with wave.open(audio_file, 'rb') as wf:
            # Get parameters
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read raw audio data
            raw_data = wf.readframes(n_frames)
            
            # Convert to numpy array using numpy's frombuffer
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert raw data to numpy array
            audio_data = np.frombuffer(raw_data, dtype=dtype)
            
            # If stereo, take the mean of the channels
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            # Calculate basic features
            if len(audio_data) == 0:
                raise ValueError("Empty audio data")
            
            # Extract statistics
            mean = np.mean(audio_data)
            std = np.std(audio_data)
            max_val = np.max(audio_data)
            min_val = np.min(audio_data)
            
            # Calculate energy
            energy = np.sum(audio_data**2) / len(audio_data)
            
            # Calculate zero crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_data))) / len(audio_data)
            
            # Extract segments and their energies
            n_segments = 10
            segment_length = len(audio_data) // n_segments
            segment_energies = []
            
            for i in range(n_segments):
                start = i * segment_length
                end = (i + 1) * segment_length if i < n_segments - 1 else len(audio_data)
                segment = audio_data[start:end]
                segment_energy = np.sum(segment**2) / len(segment) if len(segment) > 0 else 0
                segment_energies.append(segment_energy)
            
            # Create feature vector
            features = np.array([
                mean, std, max_val, min_val, energy, zero_crossings,
                *segment_energies
            ])
            
            return features
            
    except Exception as e:
        logging.error(f"Error extracting features from {audio_file}: {e}")
        # Return a zero vector if there's an error
        return np.zeros(16)

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return audio data and sample rate.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        
        raw_data = wf.readframes(n_frames)
        
        # Convert to numpy array
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert raw data to numpy array
        audio_data = np.frombuffer(raw_data, dtype=dtype)
        
        # If stereo, convert to mono
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
    return audio_data, sample_rate

def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, only log to console)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def save_to_csv(file_path: str, data: List[Tuple[str, str]]) -> None:
    """
    Save data to CSV file.
    
    Args:
        file_path: Path to save the CSV file
        data: List of tuples with (file_name, speaker)
    """
    try:
        with open(file_path, 'w') as f:
            for file_name, speaker in data:
                f.write(f"{file_name},{speaker}\n")
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")

def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True) 