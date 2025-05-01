import os
import sys
import wave
import numpy as np
import logging
import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import glob

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def extract_features(audio_file):
    """Extract basic features from an audio file"""
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

def process_dataset(data_dir):
    """Process all audio files in the dataset directory"""
    features = []
    labels = []
    file_paths = []
    
    logging.info(f"Processing audio files in {data_dir}")
    
    # Get all speaker directories
    speaker_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    speaker_dirs.sort()  # Sort to ensure consistent mapping
    
    # Create mapping from index to speaker name
    speaker_mapping = {i: name for i, name in enumerate(speaker_dirs)}
    
    # Process each speaker directory
    for idx, speaker in enumerate(speaker_dirs):
        speaker_dir = os.path.join(data_dir, speaker)
        audio_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        
        logging.info(f"Found {len(audio_files)} files for speaker {speaker}")
        
        for audio_file in audio_files:
            try:
                # Extract features
                feature_vector = extract_features(audio_file)
                
                # Add to dataset
                features.append(feature_vector)
                labels.append(idx)
                file_paths.append(audio_file)
            except Exception as e:
                logging.error(f"Failed to process {audio_file}: {e}")
    
    return np.array(features), np.array(labels), file_paths, speaker_mapping

def train_model(data_dir, output_dir="models", test_size=0.2, random_state=42):
    """Train a speaker classification model"""
    setup_logging()
    
    # Process dataset
    features, labels, file_paths, speaker_mapping = process_dataset(data_dir)
    
    if len(features) == 0:
        logging.error("No usable data extracted")
        return
    
    logging.info(f"Extracted features for {len(features)} audio files")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"Model accuracy: {accuracy:.4f}")
    logging.info(f"Macro F1 score: {f1:.4f}")
    
    # Save model and mapping
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    with open(os.path.join(output_dir, "speaker_mapping.pkl"), "wb") as f:
        pickle.dump(speaker_mapping, f)
    
    logging.info(f"Model and speaker mapping saved to {output_dir}")
    
    return model, speaker_mapping

def predict(audio_file, model, speaker_mapping):
    """Predict the speaker for a single audio file"""
    # Extract features
    features = extract_features(audio_file)
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    # Get speaker name
    speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
    
    return speaker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speaker classification model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio data")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_dir, args.test_size, args.seed) 