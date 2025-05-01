import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import argparse
import time
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

from dataset import create_dataloaders
from model import SpeakerClassifier, SpeakerClassifierWithAttention, FocalLoss
from utils import set_seed, get_device, setup_logging, calculate_metrics, plot_metrics, \
    plot_confusion_matrix, create_model_directory

def train_epoch(model: nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               criterion: nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Get batch data
        spectrograms = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = calculate_metrics(
        targets=torch.tensor(all_labels),
        predictions=torch.tensor(all_preds)
    )
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics

def validate(model: nn.Module, 
            dataloader: torch.utils.data.DataLoader, 
            criterion: nn.Module, 
            device: torch.device) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            # Get batch data
            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate validation metrics
    val_loss = running_loss / len(dataloader)
    val_metrics = calculate_metrics(
        targets=torch.tensor(all_labels),
        predictions=torch.tensor(all_preds)
    )
    val_metrics['loss'] = val_loss
    
    return val_metrics, np.array(all_labels), np.array(all_preds)

def train(args):
    """
    Train the model.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_type}_{timestamp}"
    log_dir = os.path.join("logs", run_name)
    setup_logging(log_dir)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Create model directory
    model_dir = create_model_directory(run_name)
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
    train_dataloader, val_dataloader, speaker_to_idx = create_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    num_classes = len(speaker_to_idx)
    idx_to_speaker = {idx: speaker for speaker, idx in speaker_to_idx.items()}
    
    # Save speaker mapping
    import json
    with open(os.path.join(model_dir, "speaker_mapping.json"), "w") as f:
        json.dump({"speaker_to_idx": speaker_to_idx, "idx_to_speaker": idx_to_speaker}, f)
    
    # Create model
    logging.info(f"Creating {args.model_type} model with {num_classes} classes...")
    if args.model_type == "attention":
        model = SpeakerClassifierWithAttention(num_classes=num_classes, pretrained=True)
    else:
        model = SpeakerClassifier(num_classes=num_classes, pretrained=True)
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    
    # Train the model
    logging.info(f"Starting training for {args.epochs} epochs...")
    
    best_val_f1 = 0.0
    train_metrics_list = []
    val_metrics_list = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Train for one epoch
        logging.info(f"Epoch {epoch}/{args.epochs}")
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device)
        train_metrics_list.append(train_metrics)
        
        # Validate
        val_metrics, val_labels, val_preds = validate(model, val_dataloader, criterion, device)
        val_metrics_list.append(val_metrics)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Accuracy: {train_metrics['accuracy']:.4f}, "
                    f"Train Macro F1: {train_metrics['macro_f1']:.4f}")
        logging.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                    f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        
        # Save model if validation F1 improved
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            model_path = os.path.join(model_dir, f"best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'speaker_to_idx': speaker_to_idx
            }, model_path)
            logging.info(f"Saved best model with Val Macro F1: {best_val_f1:.4f}")
            
            # Plot confusion matrix for best model
            class_names = [idx_to_speaker[idx] for idx in range(num_classes)]
            plot_confusion_matrix(val_labels, val_preds, class_names, save_dir=model_dir)
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"final_model.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': val_metrics,
        'speaker_to_idx': speaker_to_idx
    }, final_model_path)
    logging.info(f"Saved final model")
    
    # Plot metrics
    plot_metrics(train_metrics_list, val_metrics_list, save_dir=model_dir)
    
    logging.info(f"Training completed. Best validation Macro F1: {best_val_f1:.4f}")
    
    return model_dir, speaker_to_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speaker classification model")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for scheduler")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for Focal Loss")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of train samples to total")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataloaders")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--model_type", type=str, default="basic",
                        choices=["basic", "attention"],
                        help="Model type to use")
    
    args = parser.parse_args()
    train(args) 