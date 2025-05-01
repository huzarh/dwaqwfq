import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def run_command(cmd, description=None):
    """Run a shell command and print output in real-time"""
    if description:
        logging.info(description)
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.strip())
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0:
            logging.error(f"Command failed with exit code {process.returncode}")
            return False
        
        return True
    
    except Exception as e:
        logging.error(f"Error executing command: {e}")
        return False

def run_pipeline(args):
    """Run the complete speaker classification pipeline"""
    setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    models_dir = os.path.join(args.output_dir, f"models_{timestamp}")
    results_dir = os.path.join(args.output_dir, f"results_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    logging.info(f"Starting pipeline run at {timestamp}")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Models directory: {models_dir}")
    logging.info(f"Results directory: {results_dir}")
    
    # Step 1: Train model
    logging.info("--- STEP 1: Training model ---")
    train_cmd = [
        "python", "simple_extract.py",
        "--data_dir", args.data_dir,
        "--output_dir", models_dir,
        "--test_size", str(args.test_size),
        "--seed", str(args.seed)
    ]
    
    if not run_command(train_cmd, "Training speaker classification model"):
        logging.error("Training failed. Pipeline stopped.")
        return False
    
    # Step 2: Run inference (if test directory provided)
    if args.test_dir:
        logging.info("--- STEP 2: Running inference ---")
        predictions_file = os.path.join(results_dir, "predictions.csv")
        
        inference_cmd = [
            "python", "simple_predict.py",
            "--test_dir", args.test_dir,
            "--model_dir", models_dir,
            "--output_file", predictions_file
        ]
        
        if not run_command(inference_cmd, "Generating predictions for test data"):
            logging.error("Inference failed. Pipeline stopped.")
            return False
        
        logging.info(f"Predictions saved to {predictions_file}")
    
    logging.info("=== Pipeline completed successfully ===")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete speaker classification pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training dataset")
    parser.add_argument("--test_dir", type=str, help="Directory containing test audio files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base directory for outputs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    success = run_pipeline(args)
    sys.exit(0 if success else 1) 