import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def run_command(cmd, description=None):
    """Run a shell command and print its output in real-time"""
    if description:
        logging.info(description)
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Check if the command was successful
    if process.returncode != 0:
        logging.error(f"Command failed with exit code {process.returncode}")
        return False
    
    return True

def run_pipeline(args):
    """Run the entire pipeline: train model and generate predictions"""
    setup_logging()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Generate output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Pipeline started with:")
    logging.info(f"  Data directory: {args.data_dir}")
    logging.info(f"  Model directory: {args.model_dir}")
    logging.info(f"  Output directory: {output_dir}")
    
    # Step 1: Train the model
    train_cmd = [
        "python", "basic_train.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.model_dir,
        "--test_size", str(args.test_size),
        "--seed", str(args.seed)
    ]
    
    if not run_command(train_cmd, "Training model..."):
        logging.error("Training failed. Pipeline stopped.")
        return
    
    # Step 2: Run inference if test directory is provided
    if args.test_dir:
        output_file = os.path.join(output_dir, "predictions.csv")
        
        inference_cmd = [
            "python", "basic_inference.py",
            "--test_dir", args.test_dir,
            "--model_dir", args.model_dir,
            "--output_file", output_file
        ]
        
        if not run_command(inference_cmd, "Generating predictions..."):
            logging.error("Inference failed.")
            return
        
        logging.info(f"Predictions saved to {output_file}")
    
    logging.info("Pipeline completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the speaker classification pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training dataset")
    parser.add_argument("--test_dir", type=str, help="Directory containing test audio files")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    run_pipeline(args) 