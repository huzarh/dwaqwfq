import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Optional, Union, Any

def setup_logging(log_file: Optional[str] = None) -> None:
    """Set up logging configuration"""
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

def run_command(cmd: List[str], description: Optional[str] = None) -> bool:
    """
    Run a shell command and print output in real-time.
    
    Args:
        cmd: Command to run as a list of arguments
        description: Optional description of the command
        
    Returns:
        True if the command succeeded, False otherwise
    """
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

def run_training_pipeline(args: argparse.Namespace) -> bool:
    """
    Run the complete training pipeline.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if the pipeline succeeded, False otherwise
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    features_dir = os.path.join(output_dir, "features")
    model_dir = os.path.join(output_dir, "model")
    evaluation_dir = os.path.join(output_dir, "evaluation")
    log_dir = os.path.join(output_dir, "logs")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(log_dir, "pipeline.log")
    setup_logging(log_file)
    
    logging.info(f"Starting training pipeline at {timestamp}")
    logging.info(f"Data directory: {args.data_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    # Step 1: Extract features
    logging.info("=== Step 1: Extracting features ===")
    feature_extraction_cmd = [
        "python", "-m", "training.feature_extraction",
        "--data_dir", args.data_dir,
        "--output_dir", features_dir
    ]
    
    if not run_command(feature_extraction_cmd, "Extracting features from audio files"):
        logging.error("Feature extraction failed. Pipeline stopped.")
        return False
    
    # Step 2: Train model
    logging.info("=== Step 2: Training model ===")
    train_model_cmd = [
        "python", "-m", "training.train_model",
        "--data_dir", args.data_dir,
        "--output_dir", model_dir,
        "--seed", str(args.seed)
    ]
    
    if not run_command(train_model_cmd, "Training speaker classification model"):
        logging.error("Model training failed. Pipeline stopped.")
        return False
    
    # Step 3: Evaluate model (if test data is available)
    if os.path.exists(os.path.join(features_dir, "features.pkl")) and os.path.exists(os.path.join(features_dir, "labels.pkl")):
        logging.info("=== Step 3: Evaluating model ===")
        evaluate_cmd = [
            "python", "-m", "training.model_evaluation",
            "--model_dir", model_dir,
            "--test_features", os.path.join(features_dir, "features.pkl"),
            "--test_labels", os.path.join(features_dir, "labels.pkl"),
            "--output_dir", evaluation_dir
        ]
        
        if not run_command(evaluate_cmd, "Evaluating model performance"):
            logging.error("Model evaluation failed. Pipeline stopped.")
            return False
    else:
        logging.warning("No test data available for evaluation. Skipping evaluation step.")
    
    logging.info("=== Pipeline completed successfully ===")
    logging.info(f"Trained model saved to {model_dir}")
    
    if os.path.exists(evaluation_dir):
        logging.info(f"Evaluation results saved to {evaluation_dir}")
    
    return True

def main():
    """Main function for running the training pipeline"""
    parser = argparse.ArgumentParser(description="Run the complete speaker classification training pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training dataset")
    parser.add_argument("--output_dir", type=str, default="training_outputs", help="Base directory for outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    success = run_training_pipeline(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 