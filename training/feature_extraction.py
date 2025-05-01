import os
import sys
import glob
import logging
import argparse
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from training.utils import extract_features, setup_logging, create_directory

def process_audio_directory(directory: str, output_dir: Optional[str] = None) -> Tuple[np.ndarray, List[str], List[str]]:
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        logging.error(f"No WAV files found in {directory}")
        return np.array([]), [], []
    
    logging.info(f"Found {len(audio_files)} WAV files in {directory}")
    
    # ------ Özellikleri çıkarma ------ 
    features = []
    file_paths = []
    speaker_names = []
    
    for audio_file in audio_files:
        try:
            speaker_name = os.path.basename(os.path.dirname(audio_file))
            feature_vector = extract_features(audio_file)
            features.append(feature_vector)
            file_paths.append(audio_file)
            speaker_names.append(speaker_name)
            
        except Exception as e:
            logging.error(f"hatalı ses dosyası: {audio_file}: {e}")
    
    # ---> numpy array
    features_array = np.array(features)
    
    if output_dir:
        create_directory(output_dir)
        
        with open(os.path.join(output_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(features_array, f)
        
        with open(os.path.join(output_dir, 'file_paths.pkl'), 'wb') as f:
            pickle.dump(file_paths, f)
        
        with open(os.path.join(output_dir, 'speaker_names.pkl'), 'wb') as f:
            pickle.dump(speaker_names, f)
        
        logging.info(f"Saved extracted features to {output_dir}")
    
    return features_array, file_paths, speaker_names

def create_speaker_mapping(speaker_names: List[str]) -> Dict[int, str]:
    unique_speakers = sorted(set(speaker_names))
    return {i: name for i, name in enumerate(unique_speakers)}

def create_label_array(speaker_names: List[str], speaker_mapping: Dict[int, str]) -> np.ndarray:
    # ters eşleme
    reverse_mapping = {name: idx for idx, name in speaker_mapping.items()} 
    labels = np.array([reverse_mapping[name] for name in speaker_names])
    
    return labels

def main(): 
    parser = argparse.ArgumentParser(description="Extract features from audio files")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="features", help="Directory to save extracted features")
    
    args = parser.parse_args()
    
    # ------ Loglama ------ 
    setup_logging()
    
    logging.info(f"Extracting features from {args.data_dir}")
    
    # ------ Ses dosyaları işleme ------ 
    features, file_paths, speaker_names = process_audio_directory(args.data_dir, args.output_dir)
    
    if len(features) == 0:
        logging.error("No features extracted")
        return 1
    
    speaker_mapping = create_speaker_mapping(speaker_names)
        
    # ------ Ses eşleme kaydet ------ 
    with open(os.path.join(args.output_dir, 'speaker_mapping.pkl'), 'wb') as f:
        pickle.dump(speaker_mapping, f)
    
    # ------ Etiket dizisi oluştur ------ 
    labels = create_label_array(speaker_names, speaker_mapping)
    
    # ------ Etiket dizisi kaydet ------ 
    with open(os.path.join(args.output_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)
    
    logging.info(f"Çıkarılan özellikler: {len(features)} dosya")
    logging.info(f"bulunan {len(speaker_mapping)} benzersiz konuşmacı: {', '.join(speaker_mapping.values())}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 