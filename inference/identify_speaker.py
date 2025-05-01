import os
import sys
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
    find_speaker_image
)

# Import feature extraction from training module
from training.utils import extract_features

def identify_speaker(
    audio_path: str,
    model,
    speaker_mapping: Dict[int, str],
    find_image: bool = False,
    data_dir: str = "dataset"
) -> Tuple[str, Optional[str]]:
    """
    Identify the speaker of an audio file.
    
    Args:
        audio_path: Path to the audio file
        model: Trained classifier model
        speaker_mapping: Mapping from indices to speaker names
        find_image: Whether to find the speaker's image
        data_dir: Base directory containing speaker data
        
    Returns:
        Tuple of (speaker_name, image_path)
    """
    try:
        # Extract features
        features = extract_features(audio_path)
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Get speaker name
        speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
        
        # Find speaker image if requested
        image_path = None
        if find_image:
            image_path = find_speaker_image(speaker, data_dir)
        
        return speaker, image_path
    
    except Exception as e:
        logging.error(f"Error identifying speaker: {e}")
        return "unknown", None

def run_identification(args: argparse.Namespace) -> bool:
    """
    Run speaker identification for a single audio file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if successful, False otherwise
    """
    # Set up logging
    setup_logging()
    
    logging.info("Starting speaker identification")
    
    # Check if audio file exists
    if not os.path.isfile(args.audio_path):
        logging.error(f"Audio file not found: {args.audio_path}")
        return False
    
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
    
    # Identify speaker
    speaker, image_path = identify_speaker(
        args.audio_path, 
        model, 
        speaker_mapping, 
        args.find_image,
        args.data_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("SPEAKER IDENTIFICATION RESULTS")
    print("="*50)
    print(f"Audio file: {args.audio_path}")
    print(f"Identified speaker: {speaker}")
    
    if args.find_image:
        if image_path:
            print(f"Speaker image: {image_path}")
            print("\nTo view the image, you can use an image viewer or display it in a GUI application.")
        else:
            print(f"No image found for speaker {speaker}")
    
    print("="*50 + "\n")
    
    return True

def main():
    """Main function for speaker identification"""
    parser = argparse.ArgumentParser(description="Identify the speaker of an audio file")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--model_dir", type=str, help="Directory containing model and speaker mapping")
    parser.add_argument("--model_base_dir", type=str, default="training_outputs", help="Base directory for model runs")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory containing speaker data")
    parser.add_argument("--find_image", action="store_true", help="Find and display speaker's image")
    
    args = parser.parse_args()
    
    success = run_identification(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 