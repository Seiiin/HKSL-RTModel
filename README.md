# HKSL Real-Time Sign Language Recognition Model

This project implements a deep learning model for real-time Hong Kong Sign Language (HKSL) recognition using webcam input. It leverages MediaPipe for landmark extraction and TensorFlow/Keras for model building and training.

## Key Features

*   **Real-time Recognition:** Designed for live sign language translation via webcam.
*   **MediaPipe Integration:** Utilizes MediaPipe Holistic for robust extraction of pose and hand landmarks.
*   **Advanced Feature Engineering:** Extracts Position, Velocity, and Acceleration (PVA) features from landmarks (774 dimensions per frame) for improved temporal modeling.
*   **Hybrid Deep Learning Model:** Employs a combination of:
    *   1D Convolutional layers (Conv1D) for spatial feature extraction along the time axis.
    *   Bidirectional LSTMs (BiLSTM) or LSTMs to capture temporal dependencies.
    *   Multi-Head Attention (MHA) for focusing on relevant parts of the sequence.
    *   Optional CorrNet+-inspired custom layers (`AdaptedCorrelation1D`, `AdaptedTemporalWeighting1D`) for enhanced temporal correlation modeling.
*   **Configurable Pipeline:** Uses a central YAML file (`configs/config.yaml`) to manage dataset paths, model architecture, training hyperparameters, and prediction settings.
*   **Data Augmentation:** Includes noise injection, temporal masking, and spatial transformations during training to improve model robustness.
*   **Training Utilities:** Provides scripts for data recording, training with callbacks (Early Stopping, Model Checkpoint, Reduce LR), and TensorBoard logging.

## Project Structure


/Users/fanzhende/Documents/GitHub/HKSL-RTModel
├── configs/ # Configuration files (config.yaml)
│ └── config.yaml
├── dataset/ # Raw video data, organized by sign name
│ ├── sign1/
│ │ └── video1.mp4
│ └── sign2/
│ └── video2.mp4
├── hksl_rtmodel/ # Core Python package for the model
│ ├── config_loader.py # Loads configuration
│ ├── data_processing.py # Data loading, feature calculation, augmentation, preparation
│ ├── feature_extraction.py # MediaPipe landmark extraction and normalization
│ ├── models.py # Model definition (Keras)
│ ├── modules.py # Custom Keras layers (Attention Wrapper, CorrNet+ inspired)
│ ├── predictor.py # Logic for real-time prediction state management
│ ├── training.py # Training loop, callbacks, evaluation
│ └── utils.py # Utility functions (drawing landmarks, probabilities)
├── logs/ # TensorBoard logs from training runs
│ └── fit/
│ └── YYYYMMDD-HHMMSS/
├── models/ # Saved trained models (.keras) and label encoders (.pkl)
├── scripts/ # Utility scripts
│ └── record.py # Script to record new sign language samples
├── .gitignore
├── .python-version # Specifies Python version (e.g., for pyenv)
├── predict_realtime.py # Main script to run real-time prediction
├── pyproject.toml # Project metadata and dependencies (e.g., for Poetry or Hatch)
├── README.md # This file
└── train.py # Main script to train the model

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fanzhende/HKSL-RTModel.git
    cd HKSL-RTModel
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project uses `pyproject.toml`. If you use a tool like Poetry or Hatch, you can install dependencies directly:
    ```bash
    # Using Poetry
    poetry install

    # Using pip (if pyproject.toml lists dependencies under [project])
    # Make sure pip is updated: python -m pip install --upgrade pip
    pip install .

    # Or, install manually (check pyproject.toml for specific versions):
    pip install tensorflow mediapipe opencv-python scikit-learn pyyaml numpy tqdm
    ```

## Usage

### 1. Data Collection

*   Use the `record.py` script to collect video samples for different signs.
*   Run the script and follow the prompts to enter the sign name.
*   Press 's' to start/resume recording, 'q' to stop and save the current sample, and 'ESC' to exit.
*   Videos will be saved in the `dataset/` directory under subfolders named after the signs.

    ```bash
    python scripts/record.py
    ```

### 2. Configuration

*   Edit `configs/config.yaml` to adjust parameters:
    *   `data`: Dataset path, sequence length (`max_seq_length`), number of features (`num_features` - **should be 774 for PVA**), minimum video frames.
    *   `model`: Model architecture parameters (filter sizes, units, dropout, custom layer usage), model name, save directory.
    *   `training`: Epochs, batch size, learning rate, validation split, callback settings, augmentation parameters.
    *   `prediction`: Real-time prediction thresholds, buffer sizes, stability settings.
    *   `recording`: Webcam resolution for data collection.
    *   `mediapipe`: Confidence thresholds for landmark detection/tracking.

### 3. Training

*   Ensure your `dataset/` directory is populated with recorded videos.
*   Verify the `num_features` in `config.yaml` is correctly set (e.g., **774** for Position+Velocity+Acceleration).
*   Run the `train.py` script.

    ```bash
    python train.py
    ```
*   The script will:
    *   Load the configuration.
    *   Initialize MediaPipe.
    *   Process videos in the `dataset` directory to extract PVA features.
    *   Prepare data (padding, splitting).
    *   Build the model based on the config.
    *   Train the model, applying augmentations on-the-fly.
    *   Save the best model (`.keras`) and label encoder (`.pkl`) to the `models/` directory (path defined in config).
    *   Log training progress to TensorBoard in the `logs/fit` directory.
*   You can monitor training using TensorBoard:
    ```bash
    tensorboard --logdir logs/fit
    ```

### 4. Real-time Prediction

*   Ensure a trained model (`.keras`) and its corresponding label encoder (`.pkl`) exist in the `models/` directory.
*   Connect a webcam.
*   Run the `predict_realtime.py` script.

    ```bash
    python predict_realtime.py
    ```
*   The script will:
    *   Load the configuration, model, and label encoder.
    *   Initialize MediaPipe and the webcam.
    *   Continuously capture frames, extract PVA features, and feed them to the model.
    *   Apply stability logic to smooth predictions.
    *   Display the webcam feed with landmarks, the current prediction, and probability bars for top predictions.
    *   Press 'q' to quit, 'r' to reset the predictor's state.

## Model Details

*   **Input Features:** The model expects input sequences with shape `(batch_size, max_seq_length, num_features)`. For the `*_pva` models, `num_features` is 774, representing concatenated Position (258), Velocity (258), and Acceleration (258) features derived from MediaPipe landmarks.
*   **Naming Convention:** Model files and label encoders in the `models/` directory typically follow the pattern `sign_language_model_{model_name}.keras` and `label_encoder_{model_name}.pkl`, where `{model_name}` is specified in `configs/config.yaml` (e.g., `conv1d_corrplus_lstm_mha_pva`). Make sure the correct model name is configured before training or prediction.