import os
import numpy as np
import logging
import argparse
import glob
import pickle
import pandas as pd
from simple_train import extract_features, setup_logging

def load_model(model_path):
    """Load trained model."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_speaker_mapping(mapping_path):
    """Load speaker mapping."""
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)
    return mapping

def predict(model, test_dir, speaker_to_idx):
    """Generate predictions on test data."""
    # Get all test audio files
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
    file_ids = [os.path.basename(f) for f in test_files]
    
    idx_to_speaker = {idx: speaker for speaker, idx in speaker_to_idx.items()}
    
    logging.info(f"Found {len(test_files)} test files")
    
    features = []
    valid_file_ids = []
    
    # Extract features for each test file
    for file_path, file_id in zip(test_files, file_ids):
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            valid_file_ids.append(file_id)
    
    if not features:
        logging.error("No valid features extracted from test files")
        return [], []
    
    # Make predictions
    X_test = np.array(features)
    predictions = model.predict(X_test)
    
    # Convert indices to speaker names
    prediction_labels = [idx_to_speaker[idx] for idx in predictions]
    
    return valid_file_ids, prediction_labels

def run_inference(args):
    """Run inference on test data."""
    setup_logging()
    logging.info("Starting inference...")
    
    # Load model
    model_path = os.path.join(args.model_dir, "model.pkl")
    mapping_path = os.path.join(args.model_dir, "speaker_mapping.pkl")
    
    logging.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    logging.info(f"Loading speaker mapping from {mapping_path}")
    mapping = load_speaker_mapping(mapping_path)
    speaker_to_idx = mapping["speaker_to_idx"]
    
    # Generate predictions
    logging.info(f"Generating predictions on {args.test_dir}")
    file_ids, predictions = predict(model, args.test_dir, speaker_to_idx)
    
    # Save predictions to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df = pd.DataFrame({
        'sample_id': file_ids,
        'person_id': predictions
    })
    df.to_csv(args.output_file, index=False)
    
    logging.info(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions with trained model")
    parser.add_argument("--test_dir", type=str, required=True,
                      help="Path to directory with test audio files")
    parser.add_argument("--model_dir", type=str, default="models/rf_model",
                      help="Path to directory containing trained model")
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                      help="Path to output CSV file")
    
    args = parser.parse_args()
    run_inference(args) 