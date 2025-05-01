import os
import sys
import glob
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

# Add parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import from this package
from inference.utils import (
    setup_logging, 
    load_model, 
    load_speaker_mapping, 
    find_latest_model_dir,
    save_predictions_to_csv
)

# Import feature extraction from training module
from training.utils import extract_features

def collect_audio_files(directory: str) -> List[str]:
    """
    Collect all audio files from a directory.
    
    Args:
        directory: Directory to search for audio files
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        logging.error(f"No WAV files found in {directory}")
    else:
        logging.info(f"Found {len(audio_files)} audio files in {directory}")
    
    return audio_files

def predict_speakers(
    audio_files: List[str],
    model,
    speaker_mapping: Dict[int, str]
) -> List[Tuple[str, str]]:
    """
    Predict speakers for a list of audio files.
    
    Args:
        audio_files: List of audio file paths
        model: Trained classifier model
        speaker_mapping: Mapping from indices to speaker names
        
    Returns:
        List of (file_name, speaker) tuples
    """
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
            
            # Get filename only (not full path)
            file_name = os.path.basename(audio_file)
            
            # Add to predictions
            predictions.append((file_name, speaker))
            logging.info(f"Predicted speaker for {file_name}: {speaker}")
            
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
    
    return predictions

def run_inference(args: argparse.Namespace) -> bool:
    """
    Run inference on audio files using a trained model.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if successful, False otherwise
    """
    # Set up logging
    log_dir = os.path.dirname(args.output_file) if args.output_file else "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inference.log")
    setup_logging(log_file)
    
    logging.info("Starting inference process")
    
    # Get model directory
    model_dir = args.model_dir
    if not model_dir:
        logging.info("No model directory specified. Trying to find latest model.")
        model_dir = find_latest_model_dir(args.model_base_dir)
        if not model_dir:
            logging.error("Could not find a valid model directory.")
            return False
    
    logging.info(f"Using model directory: {model_dir}")
    
    # Load model and speaker mapping
    model_path = os.path.join(model_dir, "model.pkl")
    mapping_path = os.path.join(model_dir, "speaker_mapping.pkl")
    
    model = load_model(model_path)
    speaker_mapping = load_speaker_mapping(mapping_path)
    
    if model is None or not speaker_mapping:
        logging.error("Could not load model or speaker mapping")
        return False
    
    # Collect audio files
    audio_files = collect_audio_files(args.test_dir)
    
    if not audio_files:
        return False
    
    # Make predictions
    predictions = predict_speakers(audio_files, model, speaker_mapping)
    
    if not predictions:
        logging.error("No predictions generated")
        return False
    
    # Save predictions to CSV
    return save_predictions_to_csv(predictions, args.output_file)

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description="Run inference on audio files using a trained model")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test audio files")
    parser.add_argument("--model_dir", type=str, help="Directory containing model and speaker mapping")
    parser.add_argument("--model_base_dir", type=str, default="training_outputs", help="Base directory for model runs")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file for predictions")
    
    args = parser.parse_args()
    
    success = run_inference(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 