import os
import sys
import glob
import logging
import argparse
import pickle
import numpy as np
from datetime import datetime
import wave

# Import the feature extraction function from simple_extract.py
from simple_extract import extract_features  

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_model(model_path):
    """Load the trained model from the given path"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def load_speaker_mapping(mapping_path):
    """Load the speaker mapping from the given path"""
    try:
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        return mapping
    except Exception as e:
        logging.error(f"Failed to load speaker mapping: {e}")
        return None

def predict_files(test_dir, model, speaker_mapping):
    """Generate predictions for all audio files in the test directory"""
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
    
    # Generate predictions
    predictions = []
    for audio_file in audio_files:
        try:
            logging.info(f"Processing {audio_file}")
            
            # Extract features
            features = extract_features(audio_file)
            
            # Make prediction
            prediction = model.predict([features])[0]
            
            # Get speaker name
            speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
            
            # Get filename only
            file_name = os.path.basename(audio_file)
            
            predictions.append((file_name, speaker))
            logging.info(f"Predicted speaker for {file_name}: {speaker}")
            
        except Exception as e:
            logging.error(f"Failed to process {audio_file}: {e}")
    
    return predictions

def save_predictions(predictions, output_file):
    """Save predictions to a CSV file"""
    try:
        with open(output_file, 'w') as f:
            for file_name, speaker in predictions:
                f.write(f"{file_name},{speaker}\n")
        
        logging.info(f"Predictions saved to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Failed to save predictions: {e}")
        return False

def run_inference(args):
    """Run the inference pipeline"""
    setup_logging()
    
    # Load model and speaker mapping
    model_path = os.path.join(args.model_dir, "model.pkl")
    mapping_path = os.path.join(args.model_dir, "speaker_mapping.pkl")
    
    model = load_model(model_path)
    speaker_mapping = load_speaker_mapping(mapping_path)
    
    if model is None or speaker_mapping is None:
        logging.error("Cannot continue without model and speaker mapping")
        return False
    
    # Generate predictions
    predictions = predict_files(args.test_dir, model, speaker_mapping)
    
    if predictions is None or len(predictions) == 0:
        logging.error("No predictions generated")
        return False
    
    # Set output file
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"predictions_{timestamp}.csv"
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    return save_predictions(predictions, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speaker predictions for audio files")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test audio files")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model and speaker mapping")
    parser.add_argument("--output_file", type=str, help="Output CSV file for predictions")
    
    args = parser.parse_args()
    
    success = run_inference(args)
    sys.exit(0 if success else 1) 