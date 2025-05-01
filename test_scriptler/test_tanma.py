#!/usr/bin/env python
import os
import sys
import pickle
import logging
import argparse
import numpy as np
from simple_extract import extract_features

def get_latest_model_dir(base_dir="training_outputs"):
    """Find the latest model directory in the training outputs"""
    if not os.path.exists(base_dir):
        return None
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    if not run_dirs:
        return None
    
    # Sort by timestamp (newest first)
    run_dirs.sort(reverse=True)
    latest_run = os.path.join(base_dir, run_dirs[0])
    
    # Check if model directory exists
    model_dir = os.path.join(latest_run, "model")
    if os.path.exists(model_dir):
        return model_dir
    
    return None

def identify_speaker(audio_path, model_dir=None):
    """Identify the speaker of an audio file using the trained model"""
    # Find the latest model directory if not specified
    if model_dir is None:
        model_dir = get_latest_model_dir()
        if not model_dir:
            # Try legacy directory structure
            legacy_model_dirs = [d for d in os.listdir("outputs") if d.startswith("models_")]
            if legacy_model_dirs:
                legacy_model_dirs.sort(reverse=True)
                model_dir = os.path.join("outputs", legacy_model_dirs[0])
            else:
                print("Error: No trained models found")
                return 1
    
    # Load model and speaker mapping
    try:
        model_path = os.path.join(model_dir, "model.pkl")
        mapping_path = os.path.join(model_dir, "speaker_mapping.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(mapping_path, 'rb') as f:
            speaker_mapping = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Check if audio file exists
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return 1
    
    try:
        # Extract features from the audio file
        features = extract_features(audio_path)
        
        # Make a prediction with probability
        prediction = model.predict([features])[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([features])[0]
            confidence = probabilities[prediction] * 100  # Convert to percentage
        else:
            # If model doesn't support probabilities, use decision_function if available
            try:
                if hasattr(model, 'decision_function'):
                    # For SVM or similar models
                    decision_scores = model.decision_function([features])[0]
                    confidence = 50 + 50 * np.tanh(decision_scores[prediction])  # Scale to 0-100
                else:
                    confidence = None
            except:
                confidence = None
        
        # Get the speaker name
        speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
        
        # Print result
        print(f"\nAudio file: {audio_path}")
        print(f"Identified speaker: {speaker}")
        
        if confidence is not None:
            print(f"Confidence: {confidence:.2f}%")
        
        # Print all probabilities if available
        if hasattr(model, 'predict_proba'):
            print("\nAll probabilities:")
            for i, prob in enumerate(probabilities):
                if i in speaker_mapping:
                    name = speaker_mapping[i]
                    print(f"  {name}: {prob*100:.2f}%")
        
        print("")  # Empty line at the end
        
        return 0
    except Exception as e:
        print(f"Error identifying speaker: {e}")
        return 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Identify the speaker of an audio file")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--model_dir", type=str, help="Directory containing the trained model")
    parser.add_argument("--show_all", action="store_true", help="Show all prediction probabilities")
    
    args = parser.parse_args()
    
    return identify_speaker(args.audio_path, args.model_dir)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python test_tanma.py <audio_file_path> [--model_dir <model_directory>]")
        sys.exit(1)
    
    sys.exit(main()) 