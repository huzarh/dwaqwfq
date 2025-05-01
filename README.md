# Speaker Classification System

This project implements a speaker classification system that can identify speakers from audio recordings. The system uses basic audio feature extraction techniques and machine learning algorithms to classify audio samples.

## Overview

The system consists of several components:

1. **Feature Extraction**: Extracts meaningful features from audio files using basic audio processing techniques.
2. **Model Training**: Trains a RandomForest classifier on the extracted features.
3. **Inference**: Uses the trained model to predict speakers for new audio files.
4. **Pipeline**: Orchestrates the entire process from training to inference.
5. **Speaker Identification**: Identifies the speaker of a single audio file using the trained model.

## System Requirements

- Python 3.8+
- NumPy 
- scikit-learn
- Required Python packages are listed in `requirements.txt`

## Directory Structure

```
├── dataset/                  # Training dataset
│   ├── person1/              # Audio files for person1
│   │   ├── chunk_1.wav
│   │   ├── chunk_2.wav
│   │   └── ...
│   └── person2/              # Audio files for person2
│       ├── chunk_1.wav
│       ├── chunk_2.wav
│       └── ...
├── test_data/                # Test data
├── outputs/                  # Output directory
│   ├── models_TIMESTAMP/     # Trained models
│   └── results_TIMESTAMP/    # Inference results
├── simple_extract.py         # Feature extraction and training
├── simple_predict.py         # Inference script
├── run_pipeline.py           # Complete pipeline
├── identify_speaker.py       # Speaker identification script
└── quick_identify.py         # Simplified speaker identification
```

## Quick Start

### 1. Run the Complete Pipeline

To run the complete pipeline (training and inference):

```bash
python run_pipeline.py --data_dir dataset --test_dir test_data
```

This will:
1. Train a model on the data in `dataset/`
2. Generate predictions for audio files in `test_data/`
3. Save the model and predictions in timestamped directories under `outputs/`

### 2. Train a Model Only

To train a model without running inference:

```bash
python simple_extract.py --data_dir dataset --output_dir models
```

### 3. Run Inference Only

To run inference using a pre-trained model:

```bash
python simple_predict.py --test_dir test_data --model_dir models --output_file predictions.csv
```

### 4. Identify a Single Speaker

To identify the speaker of a single audio file:

```bash
python identify_speaker.py --audio_path /path/to/audio.wav --model_dir outputs/models_TIMESTAMP
```

### 5. Quick Speaker Identification

For quick identification using the latest trained model:

```bash
python quick_identify.py /path/to/audio.wav
```

## Technical Details

### Feature Extraction

The system extracts the following features from audio files:

- Basic statistical features (mean, standard deviation, min, max)
- Energy
- Zero-crossing rate
- Segmented energy features (energy in different parts of the audio)

### Model

The classification model is a RandomForest classifier with 100 trees. The model is trained on a dataset of audio samples from different speakers.

### Performance

The model achieves approximately 95% accuracy and F1 score on the test set. Performance may vary depending on the quality and quantity of training data.

## Adding New Speakers

To add new speakers to the system:

1. Create a new directory for the speaker under the `dataset/` directory
2. Place the speaker's audio files (WAV format) in the directory
3. Re-train the model using the training script or pipeline

## Troubleshooting

- **File Format Issues**: Ensure all audio files are in WAV format
- **Missing Dependencies**: Install required dependencies from `requirements.txt`
- **Performance Issues**: Increase the amount of training data or adjust model parameters

## Advanced Usage

Advanced parameters can be passed to the training script:

```bash
python simple_extract.py --data_dir dataset --output_dir models --test_size 0.2 --seed 42
```

Parameters:
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--seed`: Random seed for reproducibility (default: 42)

## Audio-Image Classification

The system can associate audio recognition results with corresponding images:

1. Each speaker folder contains both audio files (*.wav) and an image file (*.png)
2. After identifying a speaker from an audio file, the system can retrieve the associated image
3. This creates a link between audio classification and image identification

## License

This project is available under the MIT License.