import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Callback
import os
import datetime
from tqdm import tqdm  # Import tqdm
# Import custom layers explicitly for loading
from .modules import AdaptedCorrelation1D, AdaptedTemporalWeighting1D
from .models import SelfAttentionWrapper
import numpy as np
import traceback

# Custom Tqdm Callback
class TqdmProgressCallback(Callback):
    """Keras Callback using tqdm for epoch progress visualization."""

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.tqdm_bar = tqdm(range(self.epochs), desc="Training Progress", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_str = ""
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                 metrics_str += f" - {k}: {v:.4f}" # Format metrics
        self.tqdm_bar.set_postfix_str(metrics_str[3:]) # Remove leading " - "
        self.tqdm_bar.update(1) # Advance epoch progress bar

    def on_train_end(self, logs=None):
        self.tqdm_bar.close()


def setup_callbacks(config: dict) -> list:
    """Sets up Keras callbacks based on the training configuration, including TqdmProgressCallback."""
    train_cfg = config['training']
    model_cfg = config['model']
    callbacks = []

    # Add TqdmProgressCallback first
    callbacks.append(TqdmProgressCallback())

    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=train_cfg['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1 # Keep verbose for notification when stopping
    ))

    model_save_path = model_cfg['model_save_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # Silence ModelCheckpoint's per-epoch messages
    callbacks.append(ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=0 # Set to 0 to prevent messages like "val_loss improved/did not improve"
    ))

    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=train_cfg['reduce_lr_factor'],
        patience=train_cfg['reduce_lr_patience'],
        min_lr=1e-6,
        verbose=1 # Keep verbose for notification when LR changes
    ))

    log_dir_base = train_cfg['log_dir']
    log_dir = os.path.join(log_dir_base, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    callbacks.append(TensorBoard(
        log_dir=log_dir,
        histogram_freq=1 # Log histograms every epoch
    ))
    # Don't print the TensorBoard log dir message here, as tqdm might interfere initially.
    # It will be printed before training starts anyway in train_model_pipeline.
    # print(f"TensorBoard logs will be saved to: {log_dir}")

    return callbacks

def compile_model(model: tf.keras.Model, config: dict):
    """Compiles the Keras model with Adam optimizer and categorical crossentropy loss."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled.")
    model.summary() # Print model summary after compilation

def train_model_pipeline(model: tf.keras.Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, config: dict) -> tf.keras.callbacks.History:
    """Compiles, sets up callbacks, and runs the training loop using tf.data.Dataset inputs."""
    compile_model(model, config)
    callbacks = setup_callbacks(config)

    # Find TensorBoard log dir to print before tqdm starts
    log_dir = "Not Found"
    for cb in callbacks:
        if isinstance(cb, TensorBoard):
            log_dir = cb.log_dir
            break
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print("\nStarting model training (using Tqdm for progress)...")

    history = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=0 # Set to 0 to disable Keras default logging, rely on TqdmProgressCallback
    )
    print("Training finished.")
    return history

def evaluate_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray):
    """Loads the best saved model and evaluates it on the test set (numpy arrays)."""
    print(f"\nLoading best model from {model_path} for evaluation...")
    custom_objects = {
        'AdaptedCorrelation1D': AdaptedCorrelation1D,
        'AdaptedTemporalWeighting1D': AdaptedTemporalWeighting1D,
        'SelfAttentionWrapper': SelfAttentionWrapper
    }
    try:
        best_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        # Ensure test data types match model expectations
        X_test_eval = tf.cast(X_test, dtype=tf.float32)
        y_test_eval = tf.cast(y_test, dtype=tf.int32)

        print(f"Evaluating on test data with shape: X={X_test_eval.shape}, y={y_test_eval.shape}")
        loss, accuracy = best_model.evaluate(X_test_eval, y_test_eval, verbose=1)
        print(f"\nEvaluation on Test Set: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        return loss, accuracy
    except Exception as e:
        print(f"Error loading or evaluating model from {model_path}: {e}")
        traceback.print_exc()
        return None, None