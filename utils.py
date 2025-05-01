import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the device to run the model on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging for training."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

def calculate_metrics(targets: torch.Tensor, predictions: torch.Tensor) -> Dict[str, float]:
    """
    Calculate metrics for model evaluation.
    
    Args:
        targets: Ground truth labels
        predictions: Model predictions
        
    Returns:
        Dictionary with metrics
    """
    y_true = targets.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    class_f1 = f1_score(y_true, y_pred, average=None)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_f1': class_f1
    }

def plot_metrics(train_metrics: List[Dict[str, float]], 
                val_metrics: List[Dict[str, float]],
                save_dir: str = "plots") -> None:
    """
    Plot training and validation metrics.
    
    Args:
        train_metrics: List of dictionaries containing training metrics
        val_metrics: List of dictionaries containing validation metrics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    epochs = range(1, len(train_metrics) + 1)
    train_acc = [m['accuracy'] for m in train_metrics]
    train_f1 = [m['macro_f1'] for m in train_metrics]
    val_acc = [m['accuracy'] for m in val_metrics]
    val_f1 = [m['macro_f1'] for m in val_metrics]
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
    
    # Plot Macro F1
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_f1, 'b-', label='Training Macro F1')
    plt.plot(epochs, val_f1, 'r-', label='Validation Macro F1')
    plt.title('Training and Validation Macro F1')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'macro_f1.png'))
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         class_names: List[str],
                         save_dir: str = "plots") -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        class_names: List of class names
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot using matplotlib instead of seaborn
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def save_predictions_to_csv(file_ids: List[str], 
                           predictions: List[str], 
                           output_file: str = "predictions.csv") -> None:
    """
    Save predictions to CSV file.
    
    Args:
        file_ids: List of file IDs
        predictions: List of predicted speaker IDs
        output_file: Output CSV file path
    """
    df = pd.DataFrame({
        'sample_id': file_ids,
        'person_id': predictions
    })
    df.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")

def create_model_directory(model_name: str, base_dir: str = "models") -> str:
    """
    Create a directory for model checkpoints and artifacts.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory for models
        
    Returns:
        Path to the model directory
    """
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir 