import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def draw_probability_bars(image, probabilities, labels, top_n):
    """Draws horizontal bars for the top N probabilities on the image."""
    if probabilities is None or labels is None or len(probabilities) == 0 or len(labels) == 0:
        return image

    bar_area_start_x = image.shape[1] - 220 # Top-right x
    bar_area_start_y = 10                  # Top-right y
    bar_height = 20
    bar_max_width = 200
    bar_spacing = 10
    text_color = (255, 255, 255)
    bar_color = (100, 255, 100) # Light green
    bg_color_rgba = (50, 50, 50, 0.7) # Semi-transparent dark background

    probabilities = np.array(probabilities)
    # Get indices of top N probabilities
    num_to_show = min(top_n, len(probabilities))
    if num_to_show <= 0: return image
    top_indices = np.argsort(probabilities)[-num_to_show:][::-1]

    # Semi-transparent background overlay
    overlay = image.copy()
    bg_height = num_to_show * (bar_height + bar_spacing) + bar_spacing
    cv2.rectangle(overlay,
                  (bar_area_start_x, bar_area_start_y),
                  (bar_area_start_x + bar_max_width + 10, bar_area_start_y + bg_height),
                  bg_color_rgba[:3], -1) # Use RGB for rectangle

    alpha = bg_color_rgba[3] # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw bars and text
    for i, index in enumerate(top_indices):
        if index >= len(labels): continue # Safety check

        prob = probabilities[index]
        label = labels[index]
        bar_width = int(prob * bar_max_width)

        y_pos = bar_area_start_y + i * (bar_height + bar_spacing) + (bar_spacing // 2)
        x_pos = bar_area_start_x + 5

        cv2.rectangle(image, (x_pos, y_pos), (x_pos + bar_width, y_pos + bar_height), bar_color, -1)
        text = f"{label}: {prob:.2f}"
        cv2.putText(image, text, (x_pos + 5, y_pos + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return image


def draw_landmarks_on_image(image, results):
    """Draws MediaPipe pose and hand landmarks on the image."""
    drawing_spec_pose = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    drawing_spec_hands = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 255))

    # Draw Pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=drawing_spec_pose,
                              connection_drawing_spec=drawing_spec_pose)
    # Draw Hand landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=drawing_spec_hands,
                              connection_drawing_spec=drawing_spec_hands)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec=drawing_spec_hands,
                              connection_drawing_spec=drawing_spec_hands)
    return image

def draw_prediction_text(image, text):
    """Draws the predicted text on the bottom left with a background."""
    font_scale = 0.8 if len(text) > 15 else 1.0
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_w, text_h = text_size
    img_h, img_w = image.shape[:2]

    # Background rectangle for visibility
    bg_y1 = img_h - text_h - 20
    bg_y2 = img_h
    bg_x1 = 0
    bg_x2 = text_w + 20
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1) # Black background
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Put text on the blended image
    cv2.putText(image, text, (10, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    return image