#!/usr/bin/env python
import os
import sys
import pickle
from simple_extract import extract_features

def identify_speaker(audio_path):
    """Identify the speaker of an audio file using the latest trained model"""
    # Find the latest model directory
    model_dirs = [d for d in os.listdir("outputs") if d.startswith("models_")]
    if not model_dirs:
        print("Error: No trained models found in 'outputs' directory")
        return 1
    
    # Sort by timestamp (newest first)
    model_dirs.sort(reverse=True)
    model_dir = os.path.join("outputs", model_dirs[0])
    
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
        
        # Make a prediction
        prediction = model.predict([features])[0]
        
        # Get the speaker name
        speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
        
        # Print result
        print(f"\nAudio file: {audio_path}")
        print(f"Identified speaker: {speaker}\n")
        
        return 0
    except Exception as e:
        print(f"Error identifying speaker: {e}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_identify.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    sys.exit(identify_speaker(audio_path)) 