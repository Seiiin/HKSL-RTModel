import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from .feature_extraction import extract_landmarks_for_frame
from .data_processing import calculate_motion_features, POS_FEATURE_DIM
from .utils import draw_landmarks_on_image

class RealTimePredictor:
    """Handles state and logic for real-time sign language prediction using Pos/Vel/Acc features."""

    def __init__(self, model: tf.keras.Model, label_encoder, holistic: mp.solutions.holistic.Holistic, config: dict):
        self.model = model
        self.label_encoder = label_encoder
        self.holistic = holistic
        self.config = config
        self.sign_names = label_encoder.classes_
        self.num_classes = len(self.sign_names)

        pred_cfg = config['prediction']
        data_cfg = config['data']
        self.max_seq_length = data_cfg['max_seq_length']
        self.num_features = data_cfg['num_features'] # Expected total features (e.g., 774)
        if self.num_features != POS_FEATURE_DIM * 3:
             print(f"Warning: Configured num_features ({self.num_features}) doesn't match expected POS*3 ({POS_FEATURE_DIM*3}). Ensure model trained accordingly.")

        self.threshold = pred_cfg.get('threshold', 0.6)
        self.sequence_buffer_size = pred_cfg.get('sequence_buffer_size', self.max_seq_length)
        self.stability_frames = pred_cfg.get('stability_frames', 5) # Renamed from stability_threshold
        self.min_frames_for_prediction = pred_cfg.get('min_frames_for_prediction', 15)
        self.stability_confidence_threshold = pred_cfg.get('stability_confidence_threshold', self.threshold * 0.9)

        self.reset_state()

    def reset_state(self):
        """Resets the prediction buffer and state."""
        self.position_data = deque(maxlen=self.sequence_buffer_size)
        self.recent_predictions = deque(maxlen=self.stability_frames) # Stores (predicted_class_str, confidence)
        self.current_display_prediction = "Initializing..."
        self.last_confirmed_prediction = ""
        self.all_probabilities = np.zeros(self.num_classes)
        self.last_prediction_time = time.time()
        self.processing_time = 0
        print("Predictor state reset.")

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str, np.ndarray | None]:
        """Processes a single frame: extracts landmarks, calculates motion, predicts, applies stability logic."""
        start_time = time.time()

        # 1. MediaPipe Processing & Landmark Drawing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        annotated_frame = draw_landmarks_on_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results)

        # 2. Feature Extraction (Position only)
        position_landmarks = extract_landmarks_for_frame(results) # Shape (POS_FEATURE_DIM,) or None
        if position_landmarks is not None:
            self.position_data.append(position_landmarks)

        predicted_sign_current_frame = "..."
        top_confidence_current_frame = 0.0
        current_probabilities = self.all_probabilities # Use last known probabilities initially

        # 3. Prediction Logic
        if len(self.position_data) >= self.min_frames_for_prediction:
            current_position_sequence = np.array(list(self.position_data), dtype=np.float32)
            # Calculate velocity and acceleration for the current buffer
            current_velocity, current_acceleration = calculate_motion_features(current_position_sequence)
            # Combine features: (T, POS_FEATURE_DIM * 3)
            combined_sequence = np.concatenate([current_position_sequence, current_velocity, current_acceleration], axis=1)
            # Pad/truncate the combined sequence
            padded_sequence = pad_sequences([combined_sequence], maxlen=self.max_seq_length,
                                            padding='post', truncating='post', dtype='float32')

            expected_shape = (1, self.max_seq_length, self.num_features)
            if padded_sequence.shape == expected_shape:
                try:
                    # --- Prediction ---
                    current_probabilities = self.model.predict(padded_sequence, verbose=0)[0]
                    self.all_probabilities = current_probabilities # Store latest full probability distribution

                    predicted_index = np.argmax(current_probabilities)
                    top_confidence_current_frame = current_probabilities[predicted_index]

                    if top_confidence_current_frame >= self.threshold:
                        predicted_sign_current_frame = self.sign_names[predicted_index]
                    else:
                        predicted_sign_current_frame = "..."

                    # Store raw prediction and confidence for stability check
                    self.recent_predictions.append((predicted_sign_current_frame, top_confidence_current_frame))

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    predicted_sign_current_frame = "Error"
                    self.all_probabilities = np.zeros(self.num_classes)
                    self.recent_predictions.append(("Error", 0.0))
            else:
                print(f"Warning: Padded shape mismatch. Expected {expected_shape}, Got {padded_sequence.shape}")
                predicted_sign_current_frame = "Shape Err"
                self.all_probabilities = np.zeros(self.num_classes)
                self.recent_predictions.append(("Shape Err", 0.0))

            # 4. Stability Logic (Average confidence over stable window)
            stable_prediction_candidate = None
            if len(self.recent_predictions) == self.stability_frames:
                first_pred_sign = self.recent_predictions[0][0]
                # Check if the same *valid* sign was predicted consistently
                if first_pred_sign not in ["...", "Error", "Shape Err"] and \
                   all(p[0] == first_pred_sign for p in self.recent_predictions):
                    avg_confidence = np.mean([p[1] for p in self.recent_predictions])
                    # Check if average confidence meets the stability threshold
                    if avg_confidence >= self.stability_confidence_threshold:
                        stable_prediction_candidate = first_pred_sign

            # Update display text based on stability check
            current_conf_str = f" ({top_confidence_current_frame:.2f})" if predicted_sign_current_frame not in ["...", "Error", "Shape Err"] else ""

            if stable_prediction_candidate is not None:
                # Stable sign confirmed
                if stable_prediction_candidate != self.last_confirmed_prediction:
                     self.current_display_prediction = f"{stable_prediction_candidate}{current_conf_str}"
                     self.last_confirmed_prediction = stable_prediction_candidate
                     # Optional: Log confirmation
                     # print(f"Confirmed Prediction: {stable_prediction_candidate} (Avg Conf: {avg_confidence:.2f})")
                else:
                     # Still the same stable prediction
                     self.current_display_prediction = f"{stable_prediction_candidate}{current_conf_str}"
            else:
                 # No stable prediction confirmed
                 if self.last_confirmed_prediction != "":
                     # Was stable, now lost stability
                     self.current_display_prediction = f"({self.last_confirmed_prediction}) -> {predicted_sign_current_frame}{current_conf_str}"
                     self.last_confirmed_prediction = "" # Reset confirmed state
                 else:
                     # Still unstable
                     self.current_display_prediction = f"{predicted_sign_current_frame}{current_conf_str}"

        else:
            # Not enough frames
            self.current_display_prediction = f"Collecting ({len(self.position_data)}/{self.min_frames_for_prediction})"
            # Clear state if buffer isn't full enough
            if len(self.recent_predictions) > 0: self.recent_predictions.clear()
            if self.last_confirmed_prediction != "": self.last_confirmed_prediction = ""
            if np.any(self.all_probabilities): self.all_probabilities = np.zeros(self.num_classes)


        end_time = time.time()
        self.processing_time = end_time - start_time

        return annotated_frame, self.current_display_prediction, self.all_probabilities

    def get_processing_info(self):
        """Returns FPS information."""
        fps = 1.0 / self.processing_time if self.processing_time > 0 else 0
        return f"FPS: {fps:.1f}"