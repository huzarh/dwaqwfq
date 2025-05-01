import os
import sys
import pickle
import logging
import glob
from typing import Dict, List, Tuple, Optional, Union, Any

# Add parent directory to sys.path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import feature extraction from training module
from training.utils import extract_features

def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, only log to console)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)

def load_model(model_path: str):
    """
    Load a trained model from a pickle file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        return None

def load_speaker_mapping(mapping_path: str) -> Dict[int, str]:
    """
    Load the speaker mapping from a pickle file.
    
    Args:
        mapping_path: Path to the speaker mapping file
        
    Returns:
        Dictionary mapping indices to speaker names
    """
    try:
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        logging.info(f"Loaded speaker mapping from {mapping_path}")
        return mapping
    except Exception as e:
        logging.error(f"Error loading speaker mapping from {mapping_path}: {e}")
        return {}

def find_latest_model_dir(base_dir: str = "training_outputs") -> Optional[str]:
    """
    Find the most recent model directory.
    
    Args:
        base_dir: Base directory containing model runs
        
    Returns:
        Path to the most recent model directory, or None if not found
    """
    if not os.path.exists(base_dir):
        logging.error(f"Base directory {base_dir} does not exist")
        return None
    
    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not run_dirs:
        logging.error(f"No run directories found in {base_dir}")
        return None
    
    # Sort by timestamp (newest first)
    run_dirs.sort(reverse=True)
    latest_run_dir = os.path.join(base_dir, run_dirs[0])
    
    # Check if model directory exists
    model_dir = os.path.join(latest_run_dir, "model")
    if not os.path.exists(model_dir):
        logging.error(f"Model directory not found in {latest_run_dir}")
        return None
    
    return model_dir

def find_speaker_image(speaker_name: str, data_dir: str = "dataset") -> Optional[str]:
    """
    Find the image file associated with a speaker.
    
    Args:
        speaker_name: Name of the speaker
        data_dir: Base directory containing speaker data
        
    Returns:
        Path to the speaker's image file, or None if not found
    """
    speaker_dir = os.path.join(data_dir, speaker_name)
    if not os.path.isdir(speaker_dir):
        logging.error(f"Speaker directory {speaker_dir} does not exist")
        return None
    
    # Find image files (png, jpg, jpeg)
    image_files = []
    for ext in ["png", "jpg", "jpeg"]:
        image_files.extend(glob.glob(os.path.join(speaker_dir, f"*.{ext}")))
    
    if not image_files:
        logging.warning(f"No image files found for speaker {speaker_name}")
        return None
    
    return image_files[0]

def save_predictions_to_csv(predictions: List[Tuple[str, str]], output_file: str) -> bool:
    """
    Save predictions to a CSV file.
    
    Args:
        predictions: List of (file_name, speaker) tuples
        output_file: Path to the output CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            create_directory(output_dir)
        
        with open(output_file, 'w') as f:
            for file_name, speaker in predictions:
                f.write(f"{file_name},{speaker}\n")
        
        logging.info(f"Saved predictions to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving predictions to {output_file}: {e}")
        return False 