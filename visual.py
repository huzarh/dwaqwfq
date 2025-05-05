#!/usr/bin/env python3
import os
import sys
import glob
import importlib.util

# Check for required packages with proper import detection
def is_package_available(package_name):
    """Check if a package is available using importlib"""
    return importlib.util.find_spec(package_name) is not None

missing_packages = []
required_packages = ['numpy', 'matplotlib', 'sklearn', 'librosa']

for package in required_packages:
    if not is_package_available(package):
        missing_packages.append(package)

if missing_packages:
    print("Error: The following required packages are missing:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print("\nPlease install them using:")
    print("  pip install --user " + " ".join(missing_packages))
    print("\nIf you're having network issues, you can try:")
    print("  sudo apt-get install python3-numpy python3-matplotlib python3-sklearn")
    print("  pip install --user librosa")
    sys.exit(1)

# Import the packages now that we've verified they exist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import librosa

def extract_features(file_path):
    """
    Extract audio features from a wav file
    Returns feature vector and a quality score (0-1)
    """
    try:
        # Load audio file with librosa
        y, sr = librosa.load(file_path, sr=None)
        
        # Quality metrics
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Signal statistics for quality assessment
        rms = np.sqrt(np.mean(y**2))
        zero_crossings = librosa.zero_crossings(y).sum()
        zcr = zero_crossings / duration
        
        # Extract MFCC features - use fewer features if there are errors
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
        except Exception as e:
            print(f"Warning: Error extracting MFCC features: {e}")
            mfccs_mean = np.zeros(13)  # Fallback to zeros if MFCC extraction fails
        
        # Extract spectral features - use simpler features if there are errors
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            features = np.concatenate([
                mfccs_mean,
                [np.mean(spectral_centroids)],
                [np.mean(spectral_rolloff)],
                [np.mean(np.mean(spectral_contrast, axis=1))],
                [rms],
                [zcr]
            ])
        except Exception as e:
            print(f"Warning: Error extracting spectral features: {e}")
            # Fallback to simpler features if spectral extraction fails
            features = np.concatenate([
                mfccs_mean,
                [rms],
                [zcr]
            ])
        
        # Calculate quality score (0-1)
        # We'll use a simple heuristic based on RMS energy and zero-crossing rate
        energy_score = min(rms / 0.1, 1.0)  # Normalize, assuming 0.1 is a good RMS value
        
        # Check for clipping
        if np.max(np.abs(y)) > 0.95:
            energy_score *= 0.5  # Penalize for potential clipping
            
        # Signal-to-noise ratio estimation (simplified)
        noise_est = np.mean(np.abs(y[y < 0.01]))
        signal_est = np.mean(np.abs(y[y >= 0.01]))
        snr = signal_est / (noise_est + 1e-10)  # Avoid division by zero
        snr_score = min(snr / 10, 1.0)  # Normalize, assuming SNR of 10 is good
        
        # Combine scores for final quality metric
        quality_score = 0.5 * energy_score + 0.5 * snr_score
        
        return features, quality_score
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a placeholder feature vector (all zeros) with quality 0
        # Use a reasonable length based on our feature extraction
        return np.zeros(17), 0  # Adjust the size based on your feature vector length

def main():
    print("Audio PCA Visualization Tool")
    print("----------------------------")
    
    # Initialize lists to store all features and metadata
    all_features = []
    all_quality_scores = []
    all_persons = []
    all_filenames = []
    
    # Process all person directories
    dataset_path = 'dataset'
    person_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Found {len(person_dirs)} person directories")
    
    # Process each person directory
    for person_dir in person_dirs:
        person_path = os.path.join(dataset_path, person_dir)
        wav_files = glob.glob(os.path.join(person_path, "*.wav"))
        
        print(f"Processing {person_dir}: {len(wav_files)} audio files")
        
        # Process each WAV file
        for wav_file in wav_files:
            features, quality = extract_features(wav_file)
            
            if features is not None:
                all_features.append(features)
                all_quality_scores.append(quality)
                all_persons.append(person_dir)
                all_filenames.append(os.path.basename(wav_file))
    
    if not all_features:
        print("Error: No valid audio features could be extracted. Please check your audio files.")
        sys.exit(1)
    
    # Convert lists to arrays
    features_array = np.array(all_features)
    quality_array = np.array(all_quality_scores)
    
    print(f"Total processed samples: {len(features_array)}")
    print(f"Feature vector dimensions: {features_array.shape}")
    
    # Standardize features
    try:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_array)
    except Exception as e:
        print(f"Error during feature standardization: {e}")
        print("Trying to continue without standardization...")
        scaled_features = features_array
    
    # Apply PCA
    try:
        n_components = min(3, scaled_features.shape[1], scaled_features.shape[0])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_features)
        
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
        
        # Determine quality thresholds for visualization
        # We'll use the median as a threshold to divide high/low quality
        quality_threshold = np.median(quality_array)
        
        # Create a color map based on quality scores
        colors = []
        for quality in quality_array:
            if quality >= quality_threshold:
                colors.append('green')  # High quality
            else:
                colors.append('red')    # Low quality
        
        # Create 3D plot
        plt.figure(figsize=(12, 10))
        
        # Check for 3D projection support
        try:
            ax = plt.subplot(111, projection='3d')
            
            # Plot points
            scatter = ax.scatter(
                pca_result[:, 0],
                pca_result[:, 1],
                pca_result[:, 2],
                c=colors,
                alpha=0.6,
                s=40
            )
            
            # Add labels and title
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            ax.set_title('3D PCA of Audio Features Colored by Quality')
            
        except Exception as e:
            print(f"Error creating 3D plot: {e}")
            print("Falling back to 2D PCA plot...")
            
            # Fallback to 2D plot if 3D fails
            ax = plt.subplot(111)
            scatter = ax.scatter(
                pca_result[:, 0],
                pca_result[:, 1],
                c=colors,
                alpha=0.6,
                s=40
            )
            
            # Add labels for 2D plot
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_title('2D PCA of Audio Features Colored by Quality')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='High Quality',
                   markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Low Quality',
                   markerfacecolor='red', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add information about the number of high/low quality samples
        high_quality_count = sum(1 for q in quality_array if q >= quality_threshold)
        low_quality_count = len(quality_array) - high_quality_count
        quality_text = (f"High Quality Samples: {high_quality_count}\n"
                       f"Low Quality Samples: {low_quality_count}")
        plt.figtext(0.02, 0.02, quality_text, fontsize=10)
        
        # Save figure
        plt.savefig("audio_pca_visualization.png", dpi=300, bbox_inches='tight')
        
        print("PCA visualization complete. Saved to audio_pca_visualization.png")
        plt.show()
        
    except Exception as e:
        print(f"Error during PCA or visualization: {e}")
        print("Unable to complete the PCA visualization.")
        sys.exit(1)

if __name__ == "__main__":
    main()