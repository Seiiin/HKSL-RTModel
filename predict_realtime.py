import cv2
import tensorflow as tf
import pickle
import os
import sys
import warnings
import numpy as np
import traceback

from hksl_rtmodel.config_loader import load_config
from hksl_rtmodel.feature_extraction import initialize_holistic
from hksl_rtmodel.predictor import RealTimePredictor
from hksl_rtmodel.utils import draw_probability_bars, draw_prediction_text
# Explicitly import custom layers for loading
from hksl_rtmodel.modules import AdaptedCorrelation1D, AdaptedTemporalWeighting1D, AdaptedUnfoldTemporalWindows1D
from hksl_rtmodel.models import SelfAttentionWrapper

# --- Important Note ---
# Requires model trained on Pos+Vel+Acc features (e.g., 774 dimensions).
# ---

def main():
    """Main function to run the real-time prediction loop."""
    print("--- Starting Real-Time Sign Language Recognition (Pos+Vel+Acc Model) ---")

    # 1. Load Configuration
    try:
        config = load_config()
        print("Configuration loaded.")
        print(f"Expecting model trained with {config['data']['num_features']} features.")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Suppress TF/Warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.get_logger().setLevel('WARNING')
    warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')
    warnings.filterwarnings("ignore", message=".*is only supported for the square ROI.*")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    # 2. Load Model and Label Encoder
    model_cfg = config.get('model', {})
    model_path = model_cfg.get('model_save_path')
    encoder_path = model_cfg.get('label_encoder_path')

    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Train the model first.")
        return
    if not encoder_path or not os.path.exists(encoder_path):
        print(f"Error: Label encoder file not found at {encoder_path}.")
        return

    custom_objects = {
        'AdaptedCorrelation1D': AdaptedCorrelation1D,
        'AdaptedTemporalWeighting1D': AdaptedTemporalWeighting1D,
        'AdaptedUnfoldTemporalWindows1D': AdaptedUnfoldTemporalWindows1D,
        'SelfAttentionWrapper': SelfAttentionWrapper
    }

    try:
        print(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        # Verify input shape (optional but good practice)
        expected_input_shape = (None, config['data']['max_seq_length'], config['data']['num_features'])
        if model.input_shape != expected_input_shape:
            print(f"Warning: Model input shape {model.input_shape} != config {expected_input_shape}.")

        print(f"Loading label encoder: {encoder_path}")
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Model and encoder loaded. Signs: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"Error loading model or encoder: {e}")
        traceback.print_exc()
        return

    # 3. Initialize MediaPipe Holistic
    holistic_cfg = config.get('mediapipe', {})
    holistic = initialize_holistic(
        min_detection_confidence=holistic_cfg.get('min_detection_confidence', 0.5),
        min_tracking_confidence=holistic_cfg.get('min_tracking_confidence', 0.5)
    )
    print("MediaPipe Holistic initialized.")

    # 4. Initialize RealTimePredictor
    try:
        predictor = RealTimePredictor(model, label_encoder, holistic, config)
        print("RealTimePredictor initialized.")
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        holistic.close()
        return

    # 5. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        holistic.close()
        return

    rec_cfg = config.get('recording', {})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rec_cfg.get('frame_width', 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rec_cfg.get('frame_height', 480))

    pred_cfg = config.get('prediction', {})
    print(f"\nStarting prediction loop... Press 'q' to quit, 'r' to reset.")
    print(f"Confidence Threshold: {pred_cfg.get('threshold', 0.6)}")
    print(f"Stability Frames: {pred_cfg.get('stability_frames', 5)}")
    print(f"Stability Confidence Threshold: {pred_cfg.get('stability_confidence_threshold', 0.55)}")
    print("---")
    print("Display Format: 'SIGN (conf)' | '(LAST_STABLE) -> CURRENT (conf)' | 'Collecting...'")
    print("---")

    # 6. Real-time Loop
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # REMOVED: frame = cv2.flip(frame, 1) # No longer flip horizontally

            # Process frame using the predictor
            annotated_frame, display_text, probabilities = predictor.process_frame(frame)

            # Draw probability bars
            annotated_frame = draw_probability_bars(
                annotated_frame,
                probabilities,
                label_encoder.classes_,
                pred_cfg.get('top_n_visualization', 3)
            )
            # Draw the final prediction text
            annotated_frame = draw_prediction_text(annotated_frame, display_text)

            # Display FPS
            fps_text = predictor.get_processing_info()
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language Recognition - Real Time', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n'q' pressed, exiting...")
                break
            elif key == ord('r'):
                 print("\n'r' pressed, resetting predictor state...")
                 predictor.reset_state()

    finally:
        # 7. Release Resources
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("Webcam and resources released.")
        print("--- Real-Time Recognition Stopped ---")

if __name__ == "__main__":
    main()