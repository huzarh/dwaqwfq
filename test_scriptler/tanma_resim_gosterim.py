#!/usr/bin/env python
import os
import sys
import pickle
import glob
import argparse
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

def find_speaker_image(speaker_name, data_dir="dataset"):
    """Find the image file associated with a speaker"""
    speaker_dir = os.path.join(data_dir, speaker_name)
    if not os.path.isdir(speaker_dir):
        logging.warning(f"No directory found for speaker {speaker_name} in {data_dir}")
        return None
    
    # Look for PNG image files in the speaker's directory
    image_files = glob.glob(os.path.join(speaker_dir, "*.png"))
    if not image_files:
        logging.warning(f"No image files found for speaker {speaker_name}")
        return None
    
    # Return the first found image
    logging.info(f"Found image for speaker {speaker_name}: {image_files[0]}")
    return image_files[0]

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

def identify_speaker_with_image(audio_path, model_dir=None, data_dir="dataset", use_legacy=False, show_all=False):
    """Identify the speaker of an audio file and find their image"""
    # Set up logging
    setup_logging()
    
    # Check if audio file exists
    if not os.path.isfile(audio_path):
        logging.error(f"Audio file not found at {audio_path}")
        return 1
    
    # Find the latest model directory if not specified
    if model_dir is None:
        if use_legacy:
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
            probabilities = None
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
        
        # Find the speaker's image
        image_path = find_speaker_image(speaker, data_dir)
        
        # Print results
        print("\n" + "="*50)
        print(f"SPEAKER IDENTIFICATION RESULTS")
        print("="*50)
        print(f"Audio file: {audio_path}")
        print(f"Identified speaker: {speaker}")
        
        if confidence is not None:
            print(f"Confidence: {confidence:.2f}%")
        
        # Print all probabilities if requested or if confidence is low
        if probabilities is not None and (show_all or confidence < 70):
            print("\nAll speaker probabilities:")
            for i, prob in enumerate(probabilities):
                if i in speaker_mapping:
                    name = speaker_mapping[i]
                    print(f"  {name}: {prob*100:.2f}%")
        
        if image_path:
            print(f"\nSpeaker image: {image_path}")
            print("To view the image, you can use an image viewer or display it in a GUI application.")
        else:
            print(f"\nNo image found for speaker {speaker}")
        
        print("="*50 + "\n")
        
        return 0
    except Exception as e:
        logging.error(f"Error identifying speaker: {e}")
        return 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Identify the speaker of an audio file and find their image")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--model_dir", type=str, help="Directory containing the trained model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Base directory containing speaker data")
    parser.add_argument("--legacy", action="store_true", help="Use legacy directory structure (outputs/models_*)")
    parser.add_argument("--show_all", action="store_true", help="Show all prediction probabilities")
    
    args = parser.parse_args()
    
    return identify_speaker_with_image(args.audio_path, args.model_dir, args.data_dir, args.legacy, args.show_all)

if __name__ == "__main__":
    sys.exit(main()) 