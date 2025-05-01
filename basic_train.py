import os
import sys
import glob
import logging
import argparse
import numpy as np
import wave
import struct
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def extract_basic_features(file_path):
    """Extract very basic audio features without using librosa"""
    try:
        # Open the wave file
        with wave.open(file_path, 'rb') as wf:
            # Get basic parameters
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read all frames
            frames = wf.readframes(n_frames)
            
            # Convert binary data to numpy array
            if sample_width == 1:  # 8-bit samples
                fmt = f"{n_frames}B"  # unsigned char
                data = np.array(struct.unpack(fmt, frames))
            elif sample_width == 2:  # 16-bit samples
                fmt = f"{n_frames}h"  # short
                data = np.array(struct.unpack(fmt, frames))
            else:
                raise ValueError("Unsupported sample width")
            
            # Extract basic features
            # 1. Energy
            energy = np.sum(data**2) / len(data)
            
            # 2. Zero crossing rate (simplified)
            zero_crossings = np.sum(np.diff(np.signbit(data)))
            zero_crossing_rate = zero_crossings / len(data)
            
            # 3. Basic spectral features (very simplified)
            chunks = np.array_split(data, 10)  # Split into 10 chunks
            chunk_energies = [np.sum(chunk**2) / len(chunk) for chunk in chunks]
            
            # 4. Simple statistical features
            mean = np.mean(data)
            std = np.std(data)
            maximum = np.max(data)
            minimum = np.min(data)
            
            # Combine all features
            features = np.array([
                energy, 
                zero_crossing_rate, 
                mean, 
                std, 
                maximum, 
                minimum,
                *chunk_energies  # Unpack chunk energies
            ])
            
            return features
            
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {str(e)}")
        # Return a vector of zeros if there's an error
        return np.zeros(16)

def load_dataset(data_dir):
    """
    Load the dataset from the specified directory.
    Returns features, labels, and file paths.
    """
    logging.info("Loading dataset...")
    
    features = []
    labels = []
    file_paths = []
    
    # Get all speaker directories
    speaker_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    speaker_mapping = {}
    for idx, speaker in enumerate(sorted(speaker_dirs)):
        speaker_mapping[idx] = speaker
    
    # Process each speaker directory
    for idx, speaker in enumerate(sorted(speaker_dirs)):
        speaker_dir = os.path.join(data_dir, speaker)
        audio_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        
        logging.info(f"Processing {len(audio_files)} files for speaker {speaker}")
        
        for audio_file in audio_files:
            file_paths.append(audio_file)
            labels.append(idx)
            
    logging.info(f"Extracting features for {len(file_paths)} files...")
    
    # Extract features for all files
    for audio_file in file_paths:
        features.append(extract_basic_features(audio_file))
    
    return np.array(features), np.array(labels), file_paths, speaker_mapping

def train_model(args):
    """
    Train a model using the dataset from the specified directory.
    """
    setup_logging()
    
    # Load dataset
    X, y, file_paths, speaker_mapping = load_dataset(args.data_dir)
    
    if len(X) == 0:
        logging.error("No data was loaded. Check your data directory.")
        return
    
    logging.info(f"Dataset loaded with {len(X)} samples")
    logging.info(f"Feature shape: {X.shape}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, range(len(y)), test_size=args.test_size, random_state=args.seed
    )
    
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Testing set: {len(X_test)} samples")
    
    # Train the model
    logging.info("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"Model evaluation:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Macro F1 Score: {f1:.4f}")
    
    # Save the model and speaker mapping
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_path = os.path.join(args.output_dir, "model.pkl")
    mapping_path = os.path.join(args.output_dir, "speaker_mapping.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(mapping_path, 'wb') as f:
        pickle.dump(speaker_mapping, f)
        
    logging.info(f"Model and speaker mapping saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speech classification model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the audio dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    train_model(args) 