import os
import numpy as np
import logging
import argparse
import glob
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import librosa
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up logging
def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

# Extract features from audio
def extract_features(file_path, n_mfcc=40, n_mels=128):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_mean = np.mean(np.log(mel_spec + 1e-9), axis=1)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Extract spectral features
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        
        # Combine features
        features = np.concatenate([mfccs_mean, mel_spec_mean, chroma_mean, spectral_contrast_mean])
        return features
        
    except Exception as e:
        logging.error(f"Error extracting features from {file_path}: {str(e)}")
        return None

def load_dataset(data_dir):
    audio_files = []
    labels = []
    file_ids = []
    
    # Get all speaker directories
    speaker_dirs = sorted(glob.glob(os.path.join(data_dir, "person*")))
    speaker_to_idx = {os.path.basename(path): idx for idx, path in enumerate(speaker_dirs)}
    
    # Regex pattern to extract file ID
    file_id_pattern = re.compile(r'chunk_(\d+)\.wav')
    
    logging.info(f"Found {len(speaker_dirs)} speakers: {list(speaker_to_idx.keys())}")
    
    for speaker_dir in speaker_dirs:
        speaker_name = os.path.basename(speaker_dir)
        speaker_files = sorted(glob.glob(os.path.join(speaker_dir, "chunk_*.wav")))
        
        logging.info(f"Processing {speaker_name}: {len(speaker_files)} files")
        
        for file_path in speaker_files:
            # Extract file ID from filename
            file_id_match = file_id_pattern.search(os.path.basename(file_path))
            if file_id_match:
                features = extract_features(file_path)
                if features is not None:
                    file_id = f"chunk_{file_id_match.group(1)}.wav"
                    file_ids.append(file_id)
                    audio_files.append(file_path)
                    labels.append(speaker_to_idx[speaker_name])
    
    return np.array(file_ids), np.array(audio_files), np.array(labels), speaker_to_idx

def train_model(args):
    setup_logging()
    logging.info("Starting training...")
    
    # Load and preprocess data
    logging.info(f"Loading data from {args.data_dir}")
    file_ids, audio_files, labels, speaker_to_idx = load_dataset(args.data_dir)
    
    # Extract features
    logging.info("Extracting features...")
    features = []
    valid_indices = []
    
    for i, file_path in enumerate(audio_files):
        feat = extract_features(file_path)
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = labels[valid_indices]
    
    # Split data
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, np.array(valid_indices), test_size=1-args.train_ratio, random_state=args.seed
    )
    
    logging.info(f"Training with {X_train.shape[0]} samples, validating with {X_test.shape[0]} samples")
    
    # Train model
    logging.info("Training model...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        min_samples_split=2, 
        random_state=args.seed,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    logging.info(f"Train Accuracy: {train_acc:.4f}, Train Macro F1: {train_f1:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}, Test Macro F1: {test_f1:.4f}")
    
    # Save model
    model_dir = os.path.join("models", "rf_model")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # Save speaker mapping
    with open(os.path.join(model_dir, "speaker_mapping.pkl"), "wb") as f:
        pickle.dump({"speaker_to_idx": speaker_to_idx}, f)
    
    logging.info(f"Model saved to {model_dir}")
    
    return model_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speaker classification model")
    parser.add_argument("--data_dir", type=str, default="dataset",
                      help="Path to dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                      help="Ratio of train samples to total")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    train_model(args) 