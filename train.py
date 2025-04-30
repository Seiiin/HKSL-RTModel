import tensorflow as tf
import numpy as np
import os
import warnings
import sys
import traceback

# Import modules from the package
from hksl_rtmodel.config_loader import load_config
from hksl_rtmodel.feature_extraction import initialize_holistic
from hksl_rtmodel.data_processing import load_dataset, prepare_data_for_training, augment_sequence
from hksl_rtmodel.models import build_conv1d_lstm_model, SelfAttentionWrapper
from hksl_rtmodel.training import train_model_pipeline, evaluate_model
from hksl_rtmodel.modules import AdaptedCorrelation1D, AdaptedTemporalWeighting1D, AdaptedUnfoldTemporalWindows1D

# --- Important Note ---
# This script trains a model using Position, Velocity, and Acceleration features (774 dims).
# Ensure the config reflects this and that the model architecture is suitable.
# ---

# Wrapper for tf.py_function
def tf_augment_sequence(x, apply_noise_prob, apply_masking_prob, apply_spatial_prob,
                        noise_level, mask_ratio, max_angle, max_translate, max_scale_delta):
    """
    Wrapper for augment_sequence suitable for tf.py_function.
    """
    dummy_config = {
        'augmentation': {
            'apply_noise_prob': float(apply_noise_prob),
            'apply_masking_prob': float(apply_masking_prob),
            'apply_spatial_prob': float(apply_spatial_prob),
            'noise_level': float(noise_level),
            'mask_ratio': float(mask_ratio),
            'max_angle': float(max_angle),
            'max_translate': float(max_translate),
            'max_scale_delta': float(max_scale_delta)
        }
    }
    augmented_x = augment_sequence(x.numpy(), dummy_config)
    return tf.convert_to_tensor(augmented_x, dtype=tf.float32)


def main():
    """Main function to run the training pipeline."""
    print("--- Starting Sign Language Model Training Pipeline (Features: Pos+Vel+Acc) ---")
    print("*** IMPORTANT: Ensure config 'data.num_features' is updated (e.g., 774) and retrain the model! ***")

    # 1. Load Configuration
    try:
        config = load_config()
        print("Configuration loaded.")
        print(f"Expecting {config['data']['num_features']} features per frame.")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Suppress TF/Warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.get_logger().setLevel('WARNING')
    warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*is only supported for the square ROI.*")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 2. Initialize MediaPipe Holistic
    holistic_cfg = config.get('mediapipe', {})
    holistic = initialize_holistic(
        min_detection_confidence=holistic_cfg.get('min_detection_confidence', 0.5),
        min_tracking_confidence=holistic_cfg.get('min_tracking_confidence', 0.5)
    )
    print("MediaPipe Holistic initialized.")

    # 3. Load and Process Dataset
    try:
        sequences, encoded_labels, label_encoder = load_dataset(config, holistic)
        num_classes = len(label_encoder.classes_)
        print(f"Dataset loaded. Number of sequences: {len(sequences)}, Classes: {num_classes}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        return

    # 4. Prepare Data for Training
    try:
        X_train, X_val, y_train, y_val = prepare_data_for_training(sequences, encoded_labels, config)
        y_train = tf.cast(y_train, dtype=tf.int32)
        y_val = tf.cast(y_val, dtype=tf.int32)
    except ValueError as e:
        print(f"Error preparing data: {e}")
        return

    # Create tf.data Datasets with Augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))

    aug_cfg = config.get('training', {}).get('augmentation', {})
    apply_noise_prob_val = aug_cfg.get('apply_noise_prob', 0.5)
    apply_masking_prob_val = aug_cfg.get('apply_masking_prob', 0.3)
    apply_spatial_prob_val = aug_cfg.get('apply_spatial_prob', 0.5)
    noise_level_val = aug_cfg.get('noise_level', 0.01)
    mask_ratio_val = aug_cfg.get('mask_ratio', 0.1)
    max_angle_val = aug_cfg.get('max_angle', 5.0)
    max_translate_val = aug_cfg.get('max_translate', 0.05)
    max_scale_delta_val = aug_cfg.get('max_scale_delta', 0.1)

    train_dataset = train_dataset.map(lambda x, y: (
        tf.py_function(
            func=tf_augment_sequence,
            inp=[x,
                 tf.constant(apply_noise_prob_val, dtype=tf.float32),
                 tf.constant(apply_masking_prob_val, dtype=tf.float32),
                 tf.constant(apply_spatial_prob_val, dtype=tf.float32),
                 tf.constant(noise_level_val, dtype=tf.float32),
                 tf.constant(mask_ratio_val, dtype=tf.float32),
                 tf.constant(max_angle_val, dtype=tf.float32),
                 tf.constant(max_translate_val, dtype=tf.float32),
                 tf.constant(max_scale_delta_val, dtype=tf.float32)],
            Tout=tf.float32
        ), y),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Set shape explicitly after py_function
    train_dataset = train_dataset.map(lambda x, y: (tf.ensure_shape(x, X_train.shape[1:]), y))

    train_dataset = train_dataset.batch(config['training']['batch_size'])
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    print("Created tf.data pipeline for training with augmentation.")

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(config['training']['batch_size'])
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    print("Created tf.data pipeline for validation.")

    # 5. Build the Model
    data_cfg = config.get('data', {})
    input_shape = (X_train.shape[1], X_train.shape[2])
    if input_shape[0] != data_cfg.get('max_seq_length') or input_shape[1] != data_cfg.get('num_features'):
         print(f"Warning: Actual data shape {input_shape} differs from config ({data_cfg.get('max_seq_length')}, {data_cfg.get('num_features')}). Using actual shape.")

    model = build_conv1d_lstm_model(input_shape, num_classes, config)
    print(f"Model architecture built for input shape {input_shape}.")

    # 6. Train the Model
    try:
        history = train_model_pipeline(model, train_dataset, val_dataset, config)
        print("Model training completed.")
        if history and history.history:
             if 'val_accuracy' in history.history and history.history['val_accuracy']:
                 print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
             if 'val_loss' in history.history and history.history['val_loss']:
                 print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()
        return

    # 7. Final Evaluation
    model_save_path = config.get('model', {}).get('model_save_path', None)
    if model_save_path and os.path.exists(model_save_path):
         evaluate_model(model_save_path, X_val, y_val)
    else:
        print(f"Warning: Best model file not found at {model_save_path}. Skipping final evaluation.")

    holistic.close()
    print("MediaPipe resources released.")
    print("--- Training Pipeline Finished ---")

if __name__ == "__main__":
    main()