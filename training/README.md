# Speaker Classification Training Module

This directory contains scripts for training speaker classification models based on audio features.

## Components

The training module consists of the following components:

1. **Feature Extraction**: Extract features from audio files
2. **Model Training**: Train a RandomForest classifier using the extracted features
3. **Model Evaluation**: Evaluate the trained model and generate performance metrics
4. **Pipeline**: Run the complete training pipeline from feature extraction to evaluation

## Scripts

- `utils.py`: Utility functions for feature extraction and other common tasks
- `feature_extraction.py`: Extract features from audio files
- `train_model.py`: Train a speaker classification model
- `model_evaluation.py`: Evaluate model performance
- `run_pipeline.py`: Run the complete training pipeline

## Usage

### Complete Pipeline

To run the complete training pipeline:

```bash
python -m training.run_pipeline --data_dir dataset --output_dir training_outputs
```

This will:
1. Extract features from the audio files
2. Train a model
3. Evaluate the model
4. Save all outputs to timestamped directories

### Individual Steps

You can also run individual steps of the pipeline:

#### Feature Extraction

```bash
python -m training.feature_extraction --data_dir dataset --output_dir features
```

#### Model Training

```bash
python -m training.train_model --data_dir dataset --output_dir models
```

#### Model Evaluation

```bash
python -m training.model_evaluation --model_dir models --test_features features/features.pkl --test_labels features/labels.pkl --output_dir evaluation
```

## Output Structure

When running the complete pipeline, outputs are organized as follows:

```
training_outputs/
└── run_TIMESTAMP/
    ├── features/          # Extracted features
    │   ├── features.pkl
    │   ├── labels.pkl
    │   └── speaker_mapping.pkl
    ├── model/             # Trained model
    │   ├── model.pkl
    │   └── speaker_mapping.pkl
    ├── evaluation/        # Evaluation results
    │   ├── confusion_matrix.png
    │   ├── classification_report.txt
    │   └── metrics.pkl
    └── logs/              # Log files
        └── pipeline.log
```

## Feature Extraction

The feature extraction process extracts the following features from audio files:

- Basic statistical features (mean, standard deviation, min, max)
- Energy
- Zero-crossing rate
- Segmented energy features (energy in different parts of the audio)

These features are combined into a 16-dimensional feature vector for each audio file.

## Model Training

The model training process uses a RandomForest classifier with 100 trees. The data is split into training (80%) and test (20%) sets.

## Performance Metrics

The model evaluation generates the following performance metrics:

- Accuracy
- Macro F1 score
- Per-class precision, recall, and F1 score
- Confusion matrix visualization 