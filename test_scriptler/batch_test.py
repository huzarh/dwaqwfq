#!/usr/bin/env python
import os
import sys
import glob
import argparse
import logging
import pickle
import numpy as np
import time
import csv
from simple_extract import extract_features

def setup_logging(log_file=None):
    """Configure logging"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
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
        logging.debug(f"Extracting features from {audio_path}")
        features = extract_features(audio_path)
        
        # Make a prediction
        prediction = model.predict([features])[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([features])[0]
            confidence = probabilities[prediction] * 100  # Convert to percentage
            
            # Create dictionary of probabilities by speaker
            prob_by_speaker = {}
            for i, prob in enumerate(probabilities):
                if i in speaker_mapping:
                    prob_by_speaker[speaker_mapping[i]] = prob * 100  # Convert to percentage
        else:
            # If model doesn't support probabilities, use decision_function if available
            confidence = None
            probabilities = None
            prob_by_speaker = None
            try:
                if hasattr(model, 'decision_function'):
                    # For SVM or similar models
                    decision_scores = model.decision_function([features])[0]
                    confidence = 50 + 50 * np.tanh(decision_scores[prediction])  # Scale to 0-100
            except Exception as e:
                logging.debug(f"Could not get prediction confidence: {e}")
        
        # Get the speaker name
        speaker = speaker_mapping.get(prediction, f"unknown_{prediction}")
        
        return {
            "speaker": speaker, 
            "confidence": confidence,
            "probabilities": probabilities if hasattr(model, 'predict_proba') else None,
            "prob_by_speaker": prob_by_speaker
        }
    
    except Exception as e:
        logging.error(f"Error identifying speaker: {e}")
        return {"speaker": "unknown", "confidence": None, "probabilities": None, "prob_by_speaker": None}

def process_wav_directory(wav_dir, model, speaker_mapping, output_file=None, detailed=False):
    """Process all WAV files in the given directory"""
    # Get list of WAV files
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    if not wav_files:
        logging.error(f"No WAV files found in {wav_dir}")
        return False
    
    # Sort files by name
    wav_files.sort()
    
    logging.info(f"Found {len(wav_files)} WAV files in {wav_dir}")
    
    # Prepare output file if requested
    output_file_handle = None
    detailed_output_handle = None
    
    if output_file:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Output file for basic results
            output_file_handle = open(output_file, 'w', newline='')
            
            # Write header to output file
            writer = csv.writer(output_file_handle)
            writer.writerow(["filename", "speaker", "confidence"])
            
            # Output file for detailed results
            if detailed:
                detailed_file = os.path.splitext(output_file)[0] + "_detailed.csv"
                detailed_output_handle = open(detailed_file, 'w', newline='')
        except Exception as e:
            logging.error(f"Error opening output file: {e}")
            output_file = None
    
    # Print header for console output
    print("\n" + "="*70)
    print(f"{'FILENAME':<20}{'SPEAKER':<15}{'CONFIDENCE':<15}{'TIME (ms)':<10}")
    print("="*70)
    
    # Track statistics
    all_results = []
    speaker_stats = {}
    total_time = 0
    
    # Process each file
    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        
        # Time the identification process
        start_time = time.time()
        result = identify_speaker(wav_file, model, speaker_mapping)
        end_time = time.time()
        
        # Calculate processing time in milliseconds
        processing_time = (end_time - start_time) * 1000  # convert to ms
        total_time += processing_time
        
        # Format confidence for display
        confidence_str = f"{result['confidence']:.2f}%" if result['confidence'] is not None else "N/A"
        
        # Print result to console
        print(f"{filename:<20}{result['speaker']:<15}{confidence_str:<15}{processing_time:.1f} ms")
        
        # Add to statistics
        all_results.append({
            "filename": filename,
            "speaker": result['speaker'],
            "confidence": result['confidence'],
            "processing_time": processing_time,
            "prob_by_speaker": result['prob_by_speaker']
        })
        
        # Update speaker statistics
        if result['speaker'] not in speaker_stats:
            speaker_stats[result['speaker']] = []
        speaker_stats[result['speaker']].append({
            "filename": filename,
            "confidence": result['confidence']
        })
        
        # Save to output file if requested
        if output_file_handle:
            confidence_val = f"{result['confidence']:.2f}" if result['confidence'] is not None else "N/A"
            writer = csv.writer(output_file_handle)
            writer.writerow([filename, result['speaker'], confidence_val])
        
        # Save detailed probabilities if requested
        if detailed and detailed_output_handle and result['prob_by_speaker']:
            # Get header if first row
            if detailed_output_handle.tell() == 0:
                # Sort speaker names for consistent columns
                speaker_names = sorted(result['prob_by_speaker'].keys())
                header = ["filename"] + speaker_names
                writer = csv.writer(detailed_output_handle)
                writer.writerow(header)
            
            # Write probabilities
            row = [filename]
            for speaker in sorted(result['prob_by_speaker'].keys()):
                row.append(f"{result['prob_by_speaker'][speaker]:.2f}")
            
            writer = csv.writer(detailed_output_handle)
            writer.writerow(row)
    
    print("="*70)
    
    # Calculate and show summary statistics
    avg_time = total_time / len(wav_files) if wav_files else 0
    print(f"\nSUMMARY:")
    print(f"Total files processed: {len(wav_files)}")
    print(f"Average processing time: {avg_time:.1f} ms per file")
    
    # Show speaker distribution
    print("\nSPEAKER DISTRIBUTION:")
    for speaker, files in sorted(speaker_stats.items()):
        avg_conf = sum(f['confidence'] for f in files if f['confidence'] is not None) / len(files) if files else 0
        print(f"  {speaker}: {len(files)} files (avg confidence: {avg_conf:.2f}%)")
    
    # Close output files if opened
    if output_file_handle:
        output_file_handle.close()
        logging.info(f"Results saved to {output_file}")
    
    if detailed_output_handle:
        detailed_output_handle.close()
        logging.info(f"Detailed results saved to {os.path.splitext(output_file)[0] + '_detailed.csv'}")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process multiple audio files for speaker identification")
    parser.add_argument("--wav_dir", type=str, default="test_input/random_set",
                       help="Directory containing WAV files to process")
    parser.add_argument("--model_dir", type=str,
                       help="Directory containing the trained model (defaults to latest model)")
    parser.add_argument("--output_file", type=str,
                       help="CSV file to save results (optional)")
    parser.add_argument("--detailed", action="store_true",
                       help="Generate detailed probability output for each speaker")
    parser.add_argument("--legacy", action="store_true",
                       help="Use legacy directory structure (outputs/models_*)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed logging information")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(levelname)s - %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S')
    
    # Check if WAV directory exists
    if not os.path.isdir(args.wav_dir):
        logging.error(f"WAV directory not found: {args.wav_dir}")
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
    
    # Process WAV files
    if not process_wav_directory(args.wav_dir, model, speaker_mapping, args.output_file, args.detailed):
        logging.error("Failed to process WAV directory")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 