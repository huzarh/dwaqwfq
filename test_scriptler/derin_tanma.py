import os
import sys
import argparse
import pickle
import logging
from simple_extract import extract_features

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_model(model_dir):
    """Load the trained model and speaker mapping"""
    try:
        model_path = os.path.join(model_dir, "model.pkl")
        mapping_path = os.path.join(model_dir, "speaker_mapping.pkl")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
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
        features = extract_features(audio_path)
        
        # Make a prediction
        prediction = model.predict([features])[0]
        
        # Get the speaker name
        speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
        
        return speaker
    
    except Exception as e:
        logging.error(f"Error identifying speaker: {e}")
        return "unknown"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Identify the speaker of an audio file")
    parser.add_argument("--audio_path", type=str, required=True, 
                       help="Path to the audio file")
    parser.add_argument("--model_dir", type=str, default="outputs/models_20250501_083250", 
                       help="Directory containing the trained model")
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.isfile(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        return 1
    
    setup_logging()
    
    # Load model and speaker mapping
    model, speaker_mapping = load_model(args.model_dir)
    
    if model is None or speaker_mapping is None:
        print("Error: Failed to load model or speaker mapping")
        return 1
    
    # Identify speaker
    speaker = identify_speaker(args.audio_path, model, speaker_mapping)
    
    # Print result
    print(f"\nAudio file: {args.audio_path}")
    print(f"Identified speaker: {speaker}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 