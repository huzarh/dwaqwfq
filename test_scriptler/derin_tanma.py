import os
import sys
import argparse
import pickle
import logging
import numpy as np
from simple_extract import extract_features

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def find_latest_model_dir(base_dir="training_outputs"):
    """Find the latest model directory in the training outputs"""
    if not os.path.exists(base_dir):
        logging.warning(f"Base directory {base_dir} not found")
        return None
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    if not run_dirs:
        logging.warning(f"No run directories found in {base_dir}")
        return None
    
    # Sort by timestamp (newest first)
    run_dirs.sort(reverse=True)
    latest_run = os.path.join(base_dir, run_dirs[0])
    
    # Check if model directory exists
    model_dir = os.path.join(latest_run, "model")
    if os.path.exists(model_dir):
        logging.info(f"Found latest model directory: {model_dir}")
        return model_dir
    
    logging.warning(f"No model directory found in {latest_run}")
    return None

def load_model(model_dir):
    """Load the trained model and speaker mapping"""
    try:
        model_path = os.path.join(model_dir, "model.pkl")
        mapping_path = os.path.join(model_dir, "speaker_mapping.pkl")
        
        logging.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logging.info(f"Loading speaker mapping from {mapping_path}")
        with open(mapping_path, 'rb') as f:
            speaker_mapping = pickle.load(f)
        
        return model, speaker_mapping
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None

def identify_speaker(audio_path, model, speaker_mapping):
    """Identify the speaker of an audio file"""
    try:
        # Extract features from the audio file
        logging.info(f"Extracting features from {audio_path}")
        features = extract_features(audio_path)
        
        # Make a prediction
        logging.info("Predicting speaker")
        prediction = model.predict([features])[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            logging.info("Getting prediction probabilities")
            probabilities = model.predict_proba([features])[0]
            confidence = probabilities[prediction] * 100  # Convert to percentage
            logging.info(f"Prediction confidence: {confidence:.2f}%")
            
            # Log all probabilities
            for i, prob in enumerate(probabilities):
                if i in speaker_mapping:
                    name = speaker_mapping[i]
                    logging.info(f"Probability for {name}: {prob*100:.2f}%")
        else:
            # If model doesn't support probabilities, use decision_function if available
            confidence = None
            try:
                if hasattr(model, 'decision_function'):
                    # For SVM or similar models
                    decision_scores = model.decision_function([features])[0]
                    confidence = 50 + 50 * np.tanh(decision_scores[prediction])  # Scale to 0-100
                    logging.info(f"Prediction confidence (from decision scores): {confidence:.2f}%")
            except Exception as e:
                logging.warning(f"Could not get prediction confidence: {e}")
        
        # Get the speaker name
        speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
        logging.info(f"Identified speaker: {speaker}")
        
        # Create result dictionary with confidence
        result = {
            "speaker": speaker,
            "confidence": confidence,
            "probabilities": probabilities if hasattr(model, 'predict_proba') else None
        }
        
        return result
    
    except Exception as e:
        logging.error(f"Error identifying speaker: {e}")
        return {"speaker": "unknown", "confidence": None, "probabilities": None}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Identify the speaker of an audio file")
    parser.add_argument("--audio_path", type=str, required=True, 
                       help="Path to the audio file")
    parser.add_argument("--model_dir", type=str, 
                       help="Directory containing the trained model (defaults to latest model)")
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy directory structure (outputs/models_*)")
    parser.add_argument("--show_all", action="store_true",
                       help="Show all prediction probabilities")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Check if audio file exists
    if not os.path.isfile(args.audio_path):
        logging.error(f"Audio file not found at {args.audio_path}")
        return 1
    
    # Find model directory if not specified
    model_dir = args.model_dir
    if model_dir is None:
        if args.legacy:
            # Try legacy directory structure
            legacy_model_dirs = [d for d in os.listdir("outputs") if d.startswith("models_")]
            if not legacy_model_dirs:
                logging.error("No trained models found in legacy 'outputs' directory")
                return 1
            
            legacy_model_dirs.sort(reverse=True)
            model_dir = os.path.join("outputs", legacy_model_dirs[0])
            logging.info(f"Using legacy model directory: {model_dir}")
        else:
            # Use new directory structure
            model_dir = find_latest_model_dir()
            if model_dir is None:
                logging.error("No trained models found in training_outputs directory")
                return 1
    
    # Load model and speaker mapping
    model, speaker_mapping = load_model(model_dir)
    
    if model is None or speaker_mapping is None:
        logging.error("Failed to load model or speaker mapping")
        return 1
    
    # Identify speaker
    result = identify_speaker(args.audio_path, model, speaker_mapping)
    
    # Print result
    print(f"\nAudio file: {args.audio_path}")
    print(f"Identified speaker: {result['speaker']}")
    
    if result['confidence'] is not None:
        print(f"Confidence: {result['confidence']:.2f}%")
    
    # Print all probabilities if available and requested
    if result['probabilities'] is not None and (args.show_all or result['confidence'] < 70):
        print("\nAll probabilities:")
        for i, prob in enumerate(result['probabilities']):
            if i in speaker_mapping:
                name = speaker_mapping[i]
                print(f"  {name}: {prob*100:.2f}%")
    
    print("")  # Empty line
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 