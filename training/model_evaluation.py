import os
import sys
import logging
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from typing import Dict, List, Tuple, Optional, Union, Any

# Add the parent directory to the path to import utils
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import utility functions
from training.utils import setup_logging, create_directory

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

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_dir: str
) -> None:
    """
    Plot and save a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        output_dir: Directory to save the plot
    """
    create_directory(output_dir)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Also save a normalized version
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.colorbar()
    plt.xticks(tick_marks, labels, rotation=45, fontsize=12)
    plt.yticks(tick_marks, labels, fontsize=12)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black",
                    fontsize=12)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300)
    plt.close()
    
    logging.info(f"Saved confusion matrices to {output_dir}")

def plot_class_metrics(
    classification_report_dict: Dict[str, Dict[str, float]],
    output_dir: str
) -> None:
    """
    Plot and save class-specific metrics.
    
    Args:
        classification_report_dict: Classification report as dictionary
        output_dir: Directory to save the plots
    """
    create_directory(output_dir)
    
    # Extract class metrics (excluding averages)
    classes = []
    precision = []
    recall = []
    f1_scores = []
    
    for label, metrics in classification_report_dict.items():
        if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
            classes.append(label)
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            f1_scores.append(metrics['f1-score'])
    
    # Plot class-specific metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1-score')
    
    plt.xlabel('Speaker', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Performance Metrics by Speaker', fontsize=16)
    plt.xticks(x, classes, rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(precision):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(recall):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(f1_scores):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'class_metrics.png'), dpi=300)
    plt.close()
    
    logging.info(f"Saved class metrics plot to {output_dir}")

def plot_feature_importance(
    model,
    output_dir: str
) -> None:
    """
    Plot and save feature importance if available in the model.
    
    Args:
        model: Trained model
        output_dir: Directory to save the plot
    """
    create_directory(output_dir)
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title('Feature Importances', fontsize=16)
            plt.bar(range(len(importances)), importances[indices], align='center')
            
            # Feature names based on our standard feature extraction
            feature_names = ['Mean', 'Std', 'Max', 'Min', 'Energy', 'ZCR']
            for i in range(10):
                feature_names.append(f'Segment {i+1} Energy')
            
            # Make sure we don't try to display more features than we have names for
            n_features = min(len(importances), len(feature_names))
            plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], 
                      rotation=90, fontsize=12)
            
            plt.xlabel('Features', fontsize=14)
            plt.ylabel('Importance', fontsize=14)
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
            plt.close()
            
            logging.info(f"Saved feature importance plot to {output_dir}")
    except Exception as e:
        logging.warning(f"Could not plot feature importance: {e}")

def plot_pca_visualization(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    speaker_mapping: Dict[int, str],
    output_dir: str
) -> None:
    """
    Plot and save a PCA visualization of the test data.
    
    Args:
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels
        speaker_mapping: Mapping from indices to speaker names
        output_dir: Directory to save the plot
    """
    create_directory(output_dir)
    
    try:
        from sklearn.decomposition import PCA
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot correctly classified points
        mask_correct = y_test == y_pred
        classes = np.unique(y_test)
        
        plt.subplot(1, 2, 1)
        for cls in classes:
            mask = (y_test == cls) & mask_correct
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=f"{speaker_mapping.get(cls, f'Speaker {cls}')} (correct)",
                       alpha=0.7)
        
        plt.title('PCA: Correctly Classified Samples', fontsize=16)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        # Plot misclassified points
        plt.subplot(1, 2, 2)
        for cls in classes:
            mask = (y_test == cls) & ~mask_correct
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       marker='x', s=100,
                       label=f"{speaker_mapping.get(cls, f'Speaker {cls}')} (misclassified)",
                       alpha=0.7)
        
        plt.title('PCA: Misclassified Samples', fontsize=16)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_visualization.png'), dpi=300)
        plt.close()
        
        logging.info(f"Saved PCA visualization to {output_dir}")
    except Exception as e:
        logging.warning(f"Could not create PCA visualization: {e}")

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    speaker_mapping: Dict[int, str],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        speaker_mapping: Mapping from indices to speaker names
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"Model accuracy: {accuracy:.4f}")
    logging.info(f"Macro F1 score: {macro_f1:.4f}")
    
    # Get speaker names for labels
    label_names = [speaker_mapping.get(i, f"Speaker {i}") for i in range(len(speaker_mapping))]
    
    # Generate and print classification report
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    logging.info("Classification Report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logging.info(f"  {label}: F1-score={metrics['f1-score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # Create visualization plots if output_dir is provided
    if output_dir:
        create_directory(output_dir)
        
        # Save classification report as text
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(classification_report(y_test, y_pred, target_names=label_names))
        
        # Save evaluation metrics as pickle
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'report': report
        }
        
        with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, label_names, output_dir)
        
        # Plot class-specific metrics
        plot_class_metrics(report, output_dir)
        
        # Plot feature importance if available
        plot_feature_importance(model, output_dir)
        
        # Plot PCA visualization
        plot_pca_visualization(X_test, y_test, y_pred, speaker_mapping, output_dir)
        
        logging.info(f"Saved evaluation results and visualizations to {output_dir}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'report': report,
        'y_pred': y_pred
    }

def main():
    """Main function for model evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate a trained speaker classification model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model and speaker mapping")
    parser.add_argument("--test_features", type=str, help="Path to test features (pickle file)")
    parser.add_argument("--test_labels", type=str, help="Path to test labels (pickle file)")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load model and speaker mapping
    model_path = os.path.join(args.model_dir, 'model.pkl')
    mapping_path = os.path.join(args.model_dir, 'speaker_mapping.pkl')
    
    model = load_model(model_path)
    speaker_mapping = load_speaker_mapping(mapping_path)
    
    if model is None or not speaker_mapping:
        logging.error("Could not load model or speaker mapping")
        return 1
    
    # Load test data
    try:
        with open(args.test_features, 'rb') as f:
            X_test = pickle.load(f)
        
        with open(args.test_labels, 'rb') as f:
            y_test = pickle.load(f)
        
        logging.info(f"Loaded test data: {len(X_test)} samples")
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return 1
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, speaker_mapping, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 