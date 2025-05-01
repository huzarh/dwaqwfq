# Speaker Classification System

This project implements a speaker classification system that can identify speakers from audio recordings. The system uses basic audio feature extraction techniques and machine learning algorithms to classify audio samples.

## Overview

The system consists of several components:

1. **Feature Extraction**: Extracts meaningful features from audio files using basic audio processing techniques.
2. **Model Training**: Trains a RandomForest classifier on the extracted features.
3. **Inference**: Uses the trained model to predict speakers for new audio files.
4. **Pipeline**: Orchestrates the entire process from training to inference.

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
├── training/                 # Training module
│   ├── utils.py              # Utility functions
│   ├── feature_extraction.py # Feature extraction
│   ├── train_model.py        # Model training
│   ├── model_evaluation.py   # Model evaluation
│   └── run_pipeline.py       # Training pipeline
├── inference/                # Inference module
│   ├── predict.py            # Prediction script
│   └── utils.py              # Inference utilities
├── simple_extract.py         # Legacy feature extraction and training
├── simple_predict.py         # Legacy inference script
└── run_pipeline.py           # Legacy complete pipeline
```

## Quick Start

### 1. Run the Complete Training Pipeline

To run the complete training pipeline:

```bash
python -m training.run_pipeline --data_dir dataset --output_dir training_outputs
```

This will:
1. Extract features from the audio files
2. Train a model
3. Evaluate the model
4. Save all outputs to timestamped directories

### 2. Run Inference on New Audio Files

To run inference using a trained model:

```bash
python simple_predict.py --test_dir test_data --model_dir training_outputs/run_TIMESTAMP/model --output_file predictions.csv
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

## Audio-Image Classification

The system can associate audio recognition results with corresponding images:

1. Each speaker folder contains both audio files (*.wav) and an image file (*.png)
2. After identifying a speaker from an audio file, the system can retrieve the associated image
3. This creates a link between audio classification and image identification

## License

This project is available under the MIT License.