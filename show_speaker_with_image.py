#!/usr/bin/env python
import os
import sys
import pickle
import glob
import argparse
from simple_extract import extract_features

def find_speaker_image(speaker_name, data_dir="dataset"):
    """Find the image file associated with a speaker"""
    speaker_dir = os.path.join(data_dir, speaker_name)
    if not os.path.isdir(speaker_dir):
        return None
    
    # Look for PNG image files in the speaker's directory
    image_files = glob.glob(os.path.join(speaker_dir, "*.png"))
    if not image_files:
        return None
    
    # Return the first found image
    return image_files[0]

def identify_speaker_with_image(audio_path, model_dir=None, data_dir="dataset"):
    """Identify the speaker of an audio file and find their image"""
    # Find the latest model directory if not specified
    if model_dir is None:
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
        
        # Find the speaker's image
        image_path = find_speaker_image(speaker, data_dir)
        
        # Print results
        print("\n" + "="*50)
        print(f"SPEAKER IDENTIFICATION RESULTS")
        print("="*50)
        print(f"Audio file: {audio_path}")
        print(f"Identified speaker: {speaker}")
        
        if image_path:
            print(f"Speaker image: {image_path}")
            print("\nTo view the image, you can use an image viewer or display it in a GUI application.")
        else:
            print(f"No image found for speaker {speaker}")
        
        print("="*50 + "\n")
        
        return 0
    except Exception as e:
        print(f"Error identifying speaker: {e}")
        return 1

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Identify the speaker of an audio file and find their image")
    parser.add_argument("audio_path", type=str, help="Path to the audio file")
    parser.add_argument("--model_dir", type=str, help="Directory containing the trained model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Base directory containing speaker data")
    
    args = parser.parse_args()
    
    return identify_speaker_with_image(args.audio_path, args.model_dir, args.data_dir)

if __name__ == "__main__":
    sys.exit(main()) 