import os
import sys
import wave
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

def extract_features(audio_file: str) -> np.ndarray:
    try:
        with wave.open(audio_file, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            raw_data = wf.readframes(n_frames)
            
            # numpy'nin frombuffer'ını kullanarak numpy dizisine dönüştürün
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # numpy dizisine dönüştür
            audio_data = np.frombuffer(raw_data, dtype=dtype)
            
            # stereo ise kanalların ortalamasını al
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            # boş ses verisi hatası
            if len(audio_data) == 0:
                raise ValueError("Empty audio data")
            
            # istatistikleri çıkar
            mean = np.mean(audio_data)
            std = np.std(audio_data)
            max_val = np.max(audio_data)
            min_val = np.min(audio_data)
            
            # enerji hesapla
            energy = np.sum(audio_data**2) / len(audio_data)
            
            # sıfır geçiş oranı hesapla
            zero_crossings = np.sum(np.diff(np.signbit(audio_data))) / len(audio_data)
            
            # segmentleri ve enerjilerini çıkar
            n_segments = 10
            segment_length = len(audio_data) // n_segments
            segment_energies = []
            
            for i in range(n_segments):
                start = i * segment_length
                end = (i + 1) * segment_length if i < n_segments - 1 else len(audio_data)
                segment = audio_data[start:end]
                segment_energy = np.sum(segment**2) / len(segment) if len(segment) > 0 else 0
                segment_energies.append(segment_energy)
            
            # özellik vektörü oluştur
            features = np.array([
                mean, std, max_val, min_val, energy, zero_crossings,
                *segment_energies
            ])
            
            return features
            
    except Exception as e:
        logging.error(f"Error extracting features from {audio_file}: {e}")
        return np.zeros(16)

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        
        raw_data = wf.readframes(n_frames)
    
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        audio_data = np.frombuffer(raw_data, dtype=dtype)
        
        # stereo ise mono'ya çevir
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
    return audio_data, sample_rate

def setup_logging(log_file: Optional[str] = None) -> None:
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
    try:
        with open(file_path, 'w') as f:
            for file_name, speaker in data:
                f.write(f"{file_name},{speaker}\n")
        logging.info(f"veri {file_path} kaydedildi")
    except Exception as e:
        logging.error(f"veri kaydedilirken hata: {file_path}: {e}")

def create_directory(directory: str) -> None:
    os.makedirs(directory, exist_ok=True) 