# Speaker Classification Inference Module

This directory contains scripts for performing inference with trained speaker classification models.

## Components

The inference module consists of the following components:

1. **Prediction**: Generate predictions for multiple audio files
2. **Speaker Identification**: Identify the speaker of a single audio file
3. **Utilities**: Common functions for model loading and feature extraction

## Scripts

- `utils.py`: Utility functions for loading models and handling predictions
- `predict.py`: Generate predictions for multiple audio files
- `identify_speaker.py`: Identify the speaker of a single audio file

## Usage

### Batch Prediction

To generate predictions for a directory of audio files:

```bash
python -m inference.predict --test_dir test_data --output_file predictions.csv
```

The script will automatically find the most recent trained model in the `training_outputs` directory. To use a specific model:

```bash
python -m inference.predict --test_dir test_data --model_dir path/to/model --output_file predictions.csv
```

### Speaker Identification

To identify the speaker of a single audio file:

```bash
python -m inference.identify_speaker path/to/audio.wav
```

To also find and display the speaker's associated image:

```bash
python -m inference.identify_speaker path/to/audio.wav --find_image
```

## Output Format

The batch prediction script outputs a CSV file with two columns:
- `filename`: The filename of the audio file (without path)
- `speaker`: The predicted speaker name

Example:
```
chunk_1.wav,person1
chunk_2.wav,person2
```

## Speaker Images

The system can associate audio recognition results with corresponding images:

1. Each speaker folder in the dataset contains both audio files (*.wav) and an image file (*.png)
2. After identifying a speaker from an audio file, the system can retrieve the associated image
3. This creates a link between audio classification and image identification

To use this functionality, run the `identify_speaker.py` script with the `--find_image` flag. 