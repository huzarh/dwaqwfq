import os
import sys
import logging
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, f1_score, classification_report
import glob

# Import the feature extraction function from the training utils
from training.utils import extract_features

def setup_logging(log_file=None):
    """Set up logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def process_dataset(data_dir):
    """Process all audio files in the dataset directory"""
    features = []
    labels = []
    file_paths = []
    
    logging.info(f"Processing audio files in {data_dir}")
    
    # Get all speaker directories
    speaker_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    speaker_dirs.sort()  # Sort to ensure consistent mapping
    
    # Create mapping from index to speaker name
    speaker_mapping = {i: name for i, name in enumerate(speaker_dirs)}
    
    # Process each speaker directory
    for idx, speaker in enumerate(speaker_dirs):
        speaker_dir = os.path.join(data_dir, speaker)
        audio_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        
        logging.info(f"Found {len(audio_files)} files for speaker {speaker}")
        
        for audio_file in audio_files:
            try:
                # Extract features
                feature_vector = extract_features(audio_file)
                
                # Add to dataset
                features.append(feature_vector)
                labels.append(idx)
                file_paths.append(audio_file)
            except Exception as e:
                logging.error(f"Failed to process {audio_file}: {e}")
    
    return np.array(features), np.array(labels), file_paths, speaker_mapping

def plot_learning_curve(estimator, X, y, title, output_dir, ylim=None, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a learning curve plot.
    
    Args:
        estimator: The model to use
        X: Features
        y: Labels
        title: Plot title
        output_dir: Directory to save the plot
        ylim: Y-axis limits
        cv: Cross-validation folds
        n_jobs: Number of jobs for parallel processing
        train_sizes: Training set sizes to plot
    """
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid(True, alpha=0.3)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300)
    plt.close()
    
    logging.info(f"Saved learning curve plot to {output_dir}")

def plot_sample_distribution(labels, speaker_mapping, output_dir):
    """
    Generate a plot showing the distribution of samples per speaker.
    
    Args:
        labels: Labels array
        speaker_mapping: Mapping from indices to speaker names
        output_dir: Directory to save the plot
    """
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Get speaker names
    speaker_names = [speaker_mapping.get(label, f"Speaker {label}") for label in unique_labels]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(speaker_names, counts, color='skyblue')
    plt.title('Number of Samples per Speaker', fontsize=16)
    plt.xlabel('Speaker', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    
    # Add the count numbers above the bars
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', fontsize=12)
    
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sample_distribution.png'), dpi=300)
    plt.close()
    
    logging.info(f"Saved sample distribution plot to {output_dir}")

def train_model(features, labels, output_dir, random_state=42):
    """Train a RandomForest model on the given features and labels"""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state
    )
    
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    logging.info(f"Model accuracy: {accuracy:.4f}")
    logging.info(f"Macro F1 score: {f1:.4f}")
    
    # Print detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logging.info("Detailed performance metrics:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logging.info(f"  Class '{label}': F1-score={metrics['f1-score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # Generate learning curve
    plot_learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=random_state),
        features, labels,
        "Learning Curve (RandomForest)",
        viz_dir
    )
    
    return model, X_test, y_test, y_pred

def save_model(model, speaker_mapping, output_dir):
    """Save the trained model and speaker mapping to the specified directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "model.pkl")
    mapping_path = os.path.join(output_dir, "speaker_mapping.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    with open(mapping_path, "wb") as f:
        pickle.dump(speaker_mapping, f)
    
    logging.info(f"Model and speaker mapping saved to {output_dir}")

def run_training(args):
    """Run the complete training pipeline"""
    # Set up logging
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    setup_logging(log_file)
    
    logging.info(f"Starting training process with data from {args.data_dir}")
    
    # Process dataset
    features, labels, file_paths, speaker_mapping = process_dataset(args.data_dir)
    
    if len(features) == 0:
        logging.error("No usable data extracted")
        return False
    
    logging.info(f"Extracted features for {len(features)} audio files")
    
    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot sample distribution
    plot_sample_distribution(labels, speaker_mapping, viz_dir)
    
    # Train model
    model, X_test, y_test, y_pred = train_model(features, labels, args.output_dir, args.seed)
    
    # Save model and mapping
    save_model(model, speaker_mapping, args.output_dir)
    
    logging.info("Training completed successfully")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speaker classification model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio data")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    success = run_training(args)
    sys.exit(0 if success else 1) 