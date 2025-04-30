import cv2
import numpy as np
import os
import pickle
import warnings
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .feature_extraction import extract_landmarks_for_frame

# --- Data Augmentation Functions ---
# Operate on sequences with shape (T, 774)
# Primarily affect the position features (first 258 columns)

POS_FEATURE_DIM = 258 # 132 pose + 63 lh + 63 rh

def add_gaussian_noise(sequence: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """Adds Gaussian noise primarily to the landmark position coordinates (x, y, z)."""
    noisy_sequence = sequence.copy()
    noise = np.random.normal(0, noise_level, (sequence.shape[0], POS_FEATURE_DIM))

    # Apply noise only to position features (first 258)
    # Pose (x, y, z) - Indices 0, 1, 2, 4, 5, 6, ... skip visibility
    for i in range(0, 33 * 4, 4):
        noisy_sequence[:, i:i+3] += noise[:, i:i+3]
    # Hands (x, y, z) - Indices 132 to 132 + 2*(21*3)
    for i in range(132, 132 + 2 * (21 * 3), 3):
         if i + 3 <= POS_FEATURE_DIM:
              noisy_sequence[:, i:i+3] += noise[:, i:i+3]
    return noisy_sequence

def random_temporal_masking(sequence: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:
    """Randomly masks entire frames (sets all features to zero for that frame)."""
    seq_len = sequence.shape[0]
    masked_sequence = sequence.copy()
    num_masked_frames = int(seq_len * mask_ratio)
    if num_masked_frames > 0 and seq_len > 0:
        masked_indices = random.sample(range(seq_len), num_masked_frames)
        masked_sequence[masked_indices, :] = 0.0
    return masked_sequence

def random_spatial_transform(sequence: np.ndarray,
                             max_angle: float = 5.0,
                             max_translate: float = 0.05,
                             max_scale_delta: float = 0.1
                             ) -> np.ndarray:
    """Applies random small rotation, translation, and scaling to normalized position landmarks."""
    transformed_sequence = sequence.copy()
    num_frames, num_total_features = sequence.shape

    if num_total_features < POS_FEATURE_DIM:
         warnings.warn("Spatial transform skipped: Not enough features for position data.")
         return sequence

    # Generate random transform parameters (same for the whole sequence)
    angle_rad = np.radians(random.uniform(-max_angle, max_angle))
    trans_x = random.uniform(-max_translate, max_translate)
    trans_y = random.uniform(-max_translate, max_translate)
    scale = random.uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta)

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Apply transformation only to position features (first POS_FEATURE_DIM columns)
    pose_feats_end = 33 * 4
    lh_feats_end = pose_feats_end + 21 * 3

    for frame_idx in range(num_frames):
        # Pose landmarks (Position only)
        for i in range(0, pose_feats_end, 4):
            xy = transformed_sequence[frame_idx, i:i+2].copy()
            rotated_scaled_xy = (rotation_matrix @ xy) * scale
            transformed_sequence[frame_idx, i:i+2] = rotated_scaled_xy + [trans_x, trans_y]
            transformed_sequence[frame_idx, i+2] *= scale # Scale Z

        # Hand landmarks (Position only)
        for i in range(pose_feats_end, pose_feats_end + 2 * (21*3), 3):
             if i+3 <= POS_FEATURE_DIM: # Ensure within position features
                 xy = transformed_sequence[frame_idx, i:i+2].copy()
                 rotated_scaled_xy = (rotation_matrix @ xy) * scale
                 transformed_sequence[frame_idx, i:i+2] = rotated_scaled_xy + [trans_x, trans_y]
                 transformed_sequence[frame_idx, i+2] *= scale # Scale Z
    # Note: Velocity/acceleration are NOT recalculated here. Assumes model learns robustness.
    return transformed_sequence

def augment_sequence(sequence: np.ndarray, config: dict) -> np.ndarray:
    """Applies a combination of augmentations stochastically."""
    aug_cfg = config.get('training', {}).get('augmentation', {})
    apply_noise_prob = aug_cfg.get('apply_noise_prob', 0.5)
    apply_masking_prob = aug_cfg.get('apply_masking_prob', 0.3)
    apply_spatial_prob = aug_cfg.get('apply_spatial_prob', 0.5)
    noise_level = aug_cfg.get('noise_level', 0.01)
    mask_ratio = aug_cfg.get('mask_ratio', 0.1)
    max_angle = aug_cfg.get('max_angle', 5.0)
    max_translate = aug_cfg.get('max_translate', 0.05)
    max_scale_delta = aug_cfg.get('max_scale_delta', 0.1)

    augmented_sequence = sequence.copy()
    if random.random() < apply_noise_prob:
        augmented_sequence = add_gaussian_noise(augmented_sequence, noise_level)
    if random.random() < apply_masking_prob:
        augmented_sequence = random_temporal_masking(augmented_sequence, mask_ratio)
    if random.random() < apply_spatial_prob:
        augmented_sequence = random_spatial_transform(augmented_sequence, max_angle, max_translate, max_scale_delta)
    return augmented_sequence

# --- Core Data Processing ---

def calculate_motion_features(position_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates velocity and acceleration from a position sequence."""
    num_frames, num_pos_features = position_sequence.shape
    velocity = np.zeros_like(position_sequence)
    acceleration = np.zeros_like(position_sequence)

    if num_frames > 1:
        velocity[1:] = position_sequence[1:] - position_sequence[:-1]
    if num_frames > 2:
        acceleration[2:] = velocity[2:] - velocity[1:-1]
    return velocity, acceleration

def extract_landmarks_from_video(video_path: str, holistic, config: dict) -> np.ndarray | None:
    """
    Extracts position landmarks, calculates velocity and acceleration,
    and concatenates them for all valid frames of a video file.
    Returns: Numpy array (num_valid_frames, num_features = 774) or None if failed.
    """
    position_landmarks_sequence = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        warnings.warn(f"Could not open video file {video_path}")
        return None

    min_frames_needed = config['data'].get('min_video_frames', 5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        frame_landmarks = extract_landmarks_for_frame(results) # Shape (258,) or None
        if frame_landmarks is not None:
            position_landmarks_sequence.append(frame_landmarks)
    cap.release()

    num_valid_frames = len(position_landmarks_sequence)
    if num_valid_frames < min_frames_needed:
        # warnings.warn(f"Video {video_path}: {num_valid_frames} valid frames < min {min_frames_needed}. Skipping.")
        return None

    position_array = np.array(position_landmarks_sequence, dtype=np.float32)
    velocity_array, acceleration_array = calculate_motion_features(position_array)

    # Concatenate features: [position, velocity, acceleration] -> (T, 774)
    combined_features = np.concatenate([position_array, velocity_array, acceleration_array], axis=1)

    expected_total_features = config['data']['num_features']
    if combined_features.shape[1] != expected_total_features:
         warnings.warn(f"Video {video_path}: combined feature count mismatch "
                       f"({combined_features.shape[1]} vs {expected_total_features}). Skipping.")
         return None
    return combined_features


def load_dataset(config: dict, holistic) -> tuple[list, list, LabelEncoder]:
    """
    Loads the dataset by processing videos, extracting position, velocity,
    and acceleration features. Augmentation applied later during training generation.
    """
    dataset_dir = config['data']['dataset_dir']
    sequences, labels = [], []
    skipped_videos = 0
    total_videos = 0
    processed_videos_in_sign = 0

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    sign_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    if not sign_names:
        raise ValueError(f"No valid sign subdirectories found in {dataset_dir}. Found: {os.listdir(dataset_dir)}")

    print(f"Loading dataset from: {dataset_dir} | Signs: {sign_names}")
    expected_num_features = config['data']['num_features'] # Should be 774

    for sign_idx, sign in enumerate(sign_names):
        sign_path = os.path.join(dataset_dir, sign)
        video_files = [f for f in os.listdir(sign_path) if f.lower().endswith(('.mp4', '.mov', '.avi')) and not f.startswith('.')]
        num_videos_in_sign = len(video_files)
        print(f"\nProcessing sign '{sign}' ({sign_idx+1}/{len(sign_names)}): {num_videos_in_sign} videos...")
        processed_videos_in_sign = 0

        for video_idx, video_file in enumerate(video_files):
            total_videos += 1
            video_path = os.path.join(sign_path, video_file)
            landmarks = extract_landmarks_from_video(video_path, holistic, config)

            if landmarks is not None:
                 if landmarks.shape[1] != expected_num_features:
                      warnings.warn(f"Video {video_path}: feature count mismatch ({landmarks.shape[1]} vs {expected_num_features}). Skipping.")
                      skipped_videos += 1
                 else:
                     sequences.append(landmarks)
                     labels.append(sign)
                     processed_videos_in_sign += 1
            else:
                skipped_videos += 1

            # Simple progress indicator within the sign folder
            print(f"\r  [{sign}] Video {video_idx+1}/{num_videos_in_sign} processed...", end="")
        print(f"\r  [{sign}] Processed {processed_videos_in_sign}/{num_videos_in_sign} videos successfully.          ") # Overwrite progress line

    print(f"\nData Loading Summary: Processed {total_videos} videos, Skipped {skipped_videos}, Collected {len(sequences)} sequences.")

    if not sequences:
        raise ValueError("No valid data sequences loaded. Check dataset, videos, and landmark extraction.")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print(f"\nEncoded labels. Classes: {len(label_encoder.classes_)} {list(label_encoder.classes_)}")

    encoder_path = config['model']['label_encoder_path']
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {encoder_path}")

    return sequences, encoded_labels, label_encoder


def prepare_data_for_training(sequences: list, encoded_labels: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pads sequences (with combined features), one-hot encodes labels, and splits data.
    """
    max_seq_length = config['data']['max_seq_length']
    num_features = config['data']['num_features'] # Should be 774
    num_classes = len(np.unique(encoded_labels))
    validation_split = config['training']['validation_split']

    sequences_np = [np.array(seq, dtype=np.float32) for seq in sequences]
    padded_sequences = pad_sequences(sequences_np, maxlen=max_seq_length, padding='post', truncating='post', dtype='float32')

    shape = padded_sequences.shape
    if len(shape) != 3 or shape[0] != len(encoded_labels):
        raise ValueError(f"Padded sequence shape {shape} mismatch with labels {len(encoded_labels)}.")
    if shape[1] != max_seq_length:
         warnings.warn(f"Padded sequence length {shape[1]} != config max {max_seq_length}.")
    if shape[2] != num_features:
        raise ValueError(f"Padded feature dimension mismatch: {shape[2]} vs expected {num_features}")

    categorical_labels = to_categorical(encoded_labels, num_classes=num_classes).astype(np.uint8)

    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, categorical_labels,
        test_size=validation_split,
        random_state=42,
        stratify=categorical_labels # Ensure class distribution is similar in train/val
    )

    print(f"\nData prepared:")
    print(f"  Train shape: {X_train.shape}, {y_train.shape}")
    print(f"  Validation shape: {X_val.shape}, {y_val.shape}")

    return X_train, X_val, y_train, y_val