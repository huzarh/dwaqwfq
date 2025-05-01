import os
import torch
import argparse
import logging
import json
from tqdm import tqdm
from typing import Dict, List, Union, Optional

from model import SpeakerClassifier, SpeakerClassifierWithAttention
from dataset import create_test_dataloader
from transforms import AudioTransforms
from utils import get_device, setup_logging, save_predictions_to_csv

def load_model(model_path: str, 
              num_classes: int = 10, 
              model_type: str = "basic", 
              device: torch.device = None) -> torch.nn.Module:
    """
    Load a trained model.
    
    Args:
        model_path: Path to saved model checkpoint
        num_classes: Number of speaker classes
        model_type: Type of the model ('basic' or 'attention')
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = get_device()
    
    # Create model architecture
    if model_type == "attention":
        model = SpeakerClassifierWithAttention(num_classes=num_classes, pretrained=False)
    else:
        model = SpeakerClassifier(num_classes=num_classes, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logging.info(f"Loaded {model_type} model from {model_path}")
    
    return model

def generate_predictions(model: torch.nn.Module, 
                        test_loader: torch.utils.data.DataLoader, 
                        idx_to_speaker: Dict[int, str],
                        device: torch.device = None) -> tuple[list[str], list[str]]:
    """
    Generate predictions on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        idx_to_speaker: Mapping from indices to speaker names
        device: Device to run inference on
        
    Returns:
        Tuple of (file_ids, predictions)
    """
    if device is None:
        device = get_device()
    
    model.eval()
    file_ids = []
    predictions = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Generating predictions")
        for batch in pbar:
            # Get batch data
            spectrograms = batch['spectrogram'].to(device)
            batch_file_ids = batch['file_id']
            
            # Forward pass
            outputs = model(spectrograms)
            _, preds = torch.max(outputs, 1)
            
            # Convert indices to speaker names
            batch_predictions = [idx_to_speaker[idx.item()] for idx in preds]
            
            # Add to results
            file_ids.extend(batch_file_ids)
            predictions.extend(batch_predictions)
    
    return file_ids, predictions

def run_inference(args):
    """
    Run inference on test data.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    setup_logging(os.path.join("logs", "inference"))
    
    # Get device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load speaker mapping
    with open(os.path.join(args.model_dir, "speaker_mapping.json"), "r") as f:
        mapping = json.load(f)
    
    speaker_to_idx = mapping["speaker_to_idx"]
    idx_to_speaker = {int(k): v for k, v in mapping["idx_to_speaker"].items()}
    num_classes = len(speaker_to_idx)
    
    # Load model
    model_path = os.path.join(args.model_dir, args.model_file)
    model = load_model(
        model_path=model_path,
        num_classes=num_classes,
        model_type=args.model_type,
        device=device
    )
    
    # Create test dataloader
    test_loader = create_test_dataloader(
        test_dir=args.test_dir,
        known_speakers=speaker_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Generate predictions
    logging.info(f"Generating predictions on {args.test_dir}...")
    file_ids, predictions = generate_predictions(
        model=model,
        test_loader=test_loader,
        idx_to_speaker=idx_to_speaker,
        device=device
    )
    
    # Save predictions to CSV
    save_predictions_to_csv(
        file_ids=file_ids,
        predictions=predictions,
        output_file=args.output_file
    )
    
    logging.info(f"Generated predictions for {len(file_ids)} files")
    logging.info(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions with trained model")
    parser.add_argument("--test_dir", type=str, required=True,
                      help="Path to directory with test audio files")
    parser.add_argument("--model_dir", type=str, required=True,
                      help="Path to directory containing trained model")
    parser.add_argument("--model_file", type=str, default="best_model.pt",
                      help="Name of model file in model_dir")
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                      help="Path to output CSV file")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for dataloaders")
    parser.add_argument("--model_type", type=str, default="basic",
                      choices=["basic", "attention"],
                      help="Model type to use")
    
    args = parser.parse_args()
    run_inference(args) 