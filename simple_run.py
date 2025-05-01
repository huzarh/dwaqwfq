import os
import argparse
import subprocess
import logging
from datetime import datetime

def run_command(command, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n=== {description} ===")
    
    print(f"Running: {command}")
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Command failed with return code {process.returncode}")
        return False
    
    return True

def run_pipeline(args):
    """Run the full pipeline."""
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("test_output", exist_ok=True)
    
    # Train model
    train_command = f"python simple_train.py --data_dir {args.data_dir} --train_ratio {args.train_ratio} --seed {args.seed}"
    
    if not run_command(train_command, "Training model"):
        print("Training failed, stopping pipeline")
        return
    
    # Generate predictions on test data
    if args.test_dir:
        test_output_dir = os.path.join("test_output", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(test_output_dir, exist_ok=True)
        output_file = os.path.join(test_output_dir, "predictions.csv")
        
        inference_command = f"python simple_inference.py --test_dir {args.test_dir} --model_dir models/rf_model --output_file {output_file}"
        
        if not run_command(inference_command, "Generating predictions"):
            print("Inference failed")
            return
    
    print("\n=== Pipeline completed successfully ===")
    print("Model saved in: models/rf_model")
    if args.test_dir:
        print(f"Predictions saved in: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the speaker classification pipeline")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="dataset",
                       help="Path to dataset directory")
    parser.add_argument("--test_dir", type=str, default="",
                       help="Path to test directory (optional)")
    
    # Training parameters
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="Ratio of train samples to total")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    run_pipeline(args) 