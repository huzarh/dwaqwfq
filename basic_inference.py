import os
import sys
import glob
import logging
import argparse
import numpy as np
import wave
import struct
import pickle
import pandas as pd
from datetime import datetime

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

def load_model(model_path):
    """Load the trained model from the given path"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return None
        
def load_speaker_mapping(mapping_path):
    """Load the speaker mapping from the given path"""
    try:
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        return mapping
    except Exception as e:
        logging.error(f"Failed to load speaker mapping: {str(e)}")
        return None

def predict(test_dir, model, speaker_mapping):
    """Generate predictions for audio files in the test directory"""
    logging.info(f"Generating predictions for audio files in {test_dir}")
    
    # Get all audio files in the test directory
    audio_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        logging.error(f"No .wav files found in {test_dir}")
        return None
        
    logging.info(f"Found {len(audio_files)} audio files")
    
    predictions = []
    
    for audio_file in audio_files:
        logging.info(f"Processing {audio_file}")
        
        # Extract features
        features = extract_basic_features(audio_file)
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Convert prediction index to speaker name
        speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
        
        # Get filename only (not full path)
        file_name = os.path.basename(audio_file)
        
        predictions.append((file_name, speaker))
    
    return predictions

def run_inference(args):
    """Run the inference pipeline"""
    setup_logging()
    
    # Load the model
    model_path = os.path.join(args.model_dir, "model.pkl")
    mapping_path = os.path.join(args.model_dir, "speaker_mapping.pkl")
    
    model = load_model(model_path)
    speaker_mapping = load_speaker_mapping(mapping_path)
    
    if model is None or speaker_mapping is None:
        logging.error("Cannot continue without model or speaker mapping")
        return
    
    # Run predictions
    predictions = predict(args.test_dir, model, speaker_mapping)
    
    if predictions is None:
        return
    
    # Save predictions to CSV
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(predictions, columns=["file", "speaker"])
    df.to_csv(output_file, index=False, header=False)
    
    logging.info(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using a trained speaker classification model")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test audio files")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model and speaker mapping")
    parser.add_argument("--output_file", type=str, help="Output CSV file for predictions")
    
    args = parser.parse_args()
    
    run_inference(args) 