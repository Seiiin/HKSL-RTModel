import mediapipe as mp
import numpy as np
import warnings

mp_holistic = mp.solutions.holistic

def initialize_holistic(min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> mp.solutions.holistic.Holistic:
    """Initializes and returns the MediaPipe Holistic model."""
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

def _normalize_pose(pose_landmarks) -> np.ndarray:
    """Normalizes pose landmarks relative to shoulder center and inter-shoulder distance. Returns (132,) features."""
    pose_features = np.zeros(33 * 4)
    if pose_landmarks is None:
        return pose_features

    try:
        p_lm = pose_landmarks.landmark
        min_vis = 0.1
        # Check visibility of key shoulder points
        if not (hasattr(p_lm[11], 'visibility') and p_lm[11].visibility > min_vis and
                hasattr(p_lm[12], 'visibility') and p_lm[12].visibility > min_vis):
            return pose_features

        shoulder_l = np.array([p_lm[11].x, p_lm[11].y, p_lm[11].z])
        shoulder_r = np.array([p_lm[12].x, p_lm[12].y, p_lm[12].z])

        center = (shoulder_l + shoulder_r) / 2
        scale = np.linalg.norm(shoulder_l - shoulder_r)
        if scale < 1e-6: scale = 1.0

        normalized_pose = []
        for lm in p_lm:
            visibility = lm.visibility if hasattr(lm, 'visibility') else 0.0
            norm_lm = (np.array([lm.x, lm.y, lm.z]) - center) / scale
            normalized_pose.extend([norm_lm[0], norm_lm[1], norm_lm[2], visibility])
        pose_features = np.array(normalized_pose).flatten()

    except IndexError:
        warnings.warn("Pose landmark index out of range. Using zeros.")
        pose_features = np.zeros(33 * 4)
    except Exception as e:
        warnings.warn(f"Error during pose normalization: {e}. Returning zeros.")
        pose_features = np.zeros(33 * 4)

    if np.any(np.isnan(pose_features)) or np.any(np.isinf(pose_features)):
        warnings.warn("NaN/Inf detected in pose features. Replacing with 0.")
        pose_features = np.nan_to_num(pose_features, nan=0.0, posinf=0.0, neginf=0.0)

    return pose_features

def _normalize_hand(hand_landmarks) -> np.ndarray:
    """Normalizes hand landmarks relative to the wrist and wrist-to-MCP distance. Returns (63,) features."""
    hand_features = np.zeros(21 * 3)
    if hand_landmarks is None or hand_landmarks.landmark is None or len(hand_landmarks.landmark) != 21:
        return hand_features

    try:
        h_lm = hand_landmarks.landmark
        wrist = np.array([h_lm[0].x, h_lm[0].y, h_lm[0].z])
        mcp_middle = np.array([h_lm[9].x, h_lm[9].y, h_lm[9].z]) # MIDDLE_FINGER_MCP

        scale = np.linalg.norm(mcp_middle - wrist)
        if scale < 1e-6:
            return hand_features # Avoid division by zero

        normalized_hand = []
        for lm in h_lm:
            norm_lm = (np.array([lm.x, lm.y, lm.z]) - wrist) / scale
            normalized_hand.extend([norm_lm[0], norm_lm[1], norm_lm[2]])
        hand_features = np.array(normalized_hand).flatten()

    except IndexError:
        warnings.warn("Hand landmark index out of range. Using zeros.")
        hand_features = np.zeros(21 * 3)
    except Exception as e:
        warnings.warn(f"Error during hand normalization: {e}. Returning zeros.")
        hand_features = np.zeros(21 * 3)

    if np.any(np.isnan(hand_features)) or np.any(np.isinf(hand_features)):
        warnings.warn("NaN or Inf detected in hand features. Replacing with 0.")
        hand_features = np.nan_to_num(hand_features, nan=0.0, posinf=0.0, neginf=0.0)

    return hand_features

def extract_landmarks_for_frame(results) -> np.ndarray | None:
    """
    Extracts and normalizes pose and hand landmarks for a SINGLE frame.
    Returns a flat numpy array of 258 position-based features or None if extraction fails.
    Shape: (258,) = 132 (pose) + 63 (lh) + 63 (rh).
    """
    if results is None:
        return None

    pose_features = _normalize_pose(results.pose_landmarks)      # Shape (132,)
    lh_features = _normalize_hand(results.left_hand_landmarks)   # Shape (63,)
    rh_features = _normalize_hand(results.right_hand_landmarks)  # Shape (63,)

    # If pose is missing and both hands are missing, frame is likely useless.
    if np.all(pose_features == 0) and np.all(lh_features == 0) and np.all(rh_features == 0):
        return None

    # Order: Pose, Left Hand, Right Hand
    frame_features = np.concatenate([pose_features, lh_features, rh_features]) # Shape (258,)

    expected_pos_features = 258
    if len(frame_features) != expected_pos_features:
         warnings.warn(f"Frame feature count mismatch: {len(frame_features)} vs {expected_pos_features}. Returning None.")
         return None

    # If all features are zero after normalization (unlikely but possible)
    if np.all(frame_features == 0):
        return None

    return frame_features.astype(np.float32)