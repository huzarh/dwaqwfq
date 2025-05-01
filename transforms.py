import os
import librosa
import numpy as np
import torch
import torchaudio
import webrtcvad
from typing import Tuple, Optional, List, Dict, Union
import random
from io import BytesIO
from pydub import AudioSegment
import logging

# Constants for audio processing
SAMPLE_RATE = 16000  # Sample rate in Hz
FRAME_DURATION_MS = 30  # Frame duration in milliseconds
N_MELS = 128  # Number of Mel bands
N_FFT = 1024  # FFT window size
HOP_LENGTH = 512  # Hop length for STFT
MAX_AUDIO_LENGTH = 4  # Maximum audio length in seconds

class AudioTransforms:
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE, 
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 apply_augmentation: bool = False,
                 vad_aggressiveness: int = 3):
        """
        Initialize audio transforms for preprocessing.
        
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of Mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
            apply_augmentation: Whether to apply data augmentation
            vad_aggressiveness: WebRTC VAD aggressiveness level (0-3)
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.apply_augmentation = apply_augmentation
        self.vad_aggressiveness = vad_aggressiveness
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Initialize mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        # Initialize augmentation transforms
        if self.apply_augmentation:
            self.time_stretch = torchaudio.transforms.TimeStretch(
                hop_length=hop_length,
                n_freq=n_fft // 2 + 1
            )
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and convert to mono.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tensor of shape (1, num_samples)
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def apply_vad(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply Voice Activity Detection to remove silence.
        
        Args:
            waveform: Audio waveform tensor of shape (1, num_samples)
            
        Returns:
            Trimmed audio waveform
        """
        # Convert to bytes for WebRTC VAD
        audio_np = waveform.numpy().flatten()
        
        # Normalize audio volume
        if np.abs(audio_np).max() > 0:
            audio_np = audio_np / np.abs(audio_np).max()
        
        # Convert to 16-bit PCM
        audio_pcm = (audio_np * 32767).astype(np.int16)
        
        # Split into frames
        samples_per_frame = int(self.sample_rate * FRAME_DURATION_MS / 1000)
        num_frames = len(audio_pcm) // samples_per_frame
        
        # Skip frames with no voice activity
        voiced_frames = []
        for i in range(num_frames):
            frame = audio_pcm[i * samples_per_frame:(i + 1) * samples_per_frame]
            frame_bytes = frame.tobytes()
            
            if len(frame_bytes) == samples_per_frame * 2:  # 2 bytes per sample for int16
                if self.vad.is_speech(frame_bytes, self.sample_rate):
                    voiced_frames.append(audio_np[i * samples_per_frame:(i + 1) * samples_per_frame])
        
        if len(voiced_frames) == 0:
            # If no voice activity detected, return the original waveform
            return waveform
        
        # Concatenate voiced frames
        voiced_audio = np.concatenate(voiced_frames)
        
        # Ensure minimum length
        if len(voiced_audio) < self.sample_rate:
            # If the voiced audio is too short, return the original waveform
            return waveform
        
        # Convert back to tensor
        return torch.from_numpy(voiced_audio).unsqueeze(0).float()
    
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-Mel spectrogram from audio waveform.
        
        Args:
            waveform: Audio waveform tensor of shape (1, num_samples)
            
        Returns:
            Log-Mel spectrogram tensor of shape (1, n_mels, time)
        """
        # Ensure input is properly sized
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply Mel spectrogram transform
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to decibel scale
        log_mel_spec = torch.log1p(mel_spec)
        
        return log_mel_spec
    
    def apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to audio waveform.
        
        Args:
            waveform: Audio waveform tensor of shape (1, num_samples)
            
        Returns:
            Augmented audio waveform
        """
        if not self.apply_augmentation:
            return waveform
        
        # Random time stretch (0.95-1.05)
        if random.random() < 0.5:
            stretch_factor = random.uniform(0.95, 1.05)
            if waveform.shape[1] > 0:
                spec = torch.stft(
                    waveform.squeeze(0),
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    window=torch.hann_window(self.n_fft),
                    return_complex=True
                )
                spec = spec.unsqueeze(0)  # Add batch dimension
                stretched_spec = self.time_stretch(spec.abs(), stretch_factor)
                # Convert back to waveform
                stretched_waveform = torch.istft(
                    stretched_spec.squeeze(0).exp().complex,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.n_fft,
                    window=torch.hann_window(self.n_fft)
                )
                waveform = stretched_waveform.unsqueeze(0)
        
        # Random pitch shift (-2 to +2 semitones)
        if random.random() < 0.5:
            waveform_np = waveform.numpy().flatten()
            pitch_shift = random.uniform(-2, 2)
            waveform_np = librosa.effects.pitch_shift(
                waveform_np, sr=self.sample_rate, n_steps=pitch_shift
            )
            waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
        
        # Add Gaussian noise
        if random.random() < 0.5:
            noise_level = random.uniform(0.001, 0.005)
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        
        return waveform
    
    def pad_or_trim(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Pad or trim spectrogram to fixed size (n_mels x n_mels).
        
        Args:
            spectrogram: Log-Mel spectrogram tensor of shape (1, n_mels, time)
            
        Returns:
            Padded/trimmed spectrogram of shape (1, n_mels, n_mels)
        """
        _, n_mels, time_length = spectrogram.shape
        
        # Target shape is (1, n_mels, n_mels)
        if time_length > n_mels:
            # Center-crop if longer
            start = (time_length - n_mels) // 2
            spectrogram = spectrogram[:, :, start:start + n_mels]
        elif time_length < n_mels:
            # Pad if shorter
            padding = n_mels - time_length
            left_pad = padding // 2
            right_pad = padding - left_pad
            spectrogram = torch.nn.functional.pad(
                spectrogram, (left_pad, right_pad, 0, 0), mode='constant'
            )
        
        return spectrogram
    
    def normalize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram to range [0, 1].
        
        Args:
            spectrogram: Log-Mel spectrogram tensor
            
        Returns:
            Normalized spectrogram
        """
        # Apply min-max normalization per channel
        min_val = spectrogram.min()
        max_val = spectrogram.max()
        
        if max_val > min_val:
            spectrogram = (spectrogram - min_val) / (max_val - min_val)
        
        return spectrogram
    
    def __call__(self, audio_path: str) -> torch.Tensor:
        """
        Apply full preprocessing pipeline on audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed spectrogram of shape (1, n_mels, n_mels)
        """
        # Load audio
        waveform = self.load_audio(audio_path)
        
        # Apply VAD
        waveform = self.apply_vad(waveform)
        
        # Apply augmentation if enabled
        if self.apply_augmentation:
            waveform = self.apply_augmentation(waveform)
        
        # Extract log-Mel spectrogram
        spectrogram = self.extract_mel_spectrogram(waveform)
        
        # Pad or trim to fixed size
        spectrogram = self.pad_or_trim(spectrogram)
        
        # Normalize
        spectrogram = self.normalize_spectrogram(spectrogram)
        
        return spectrogram 