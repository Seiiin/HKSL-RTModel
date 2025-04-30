import cv2
import os
import time
import uuid
import sys

# Ensure the main package can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from hksl_rtmodel.config_loader import load_config
except ImportError:
    print("Error: Could not import config_loader. Ensure project structure is correct.")
    sys.exit(1)

def collect_data():
    """Collects video samples for sign language gestures using webcam."""
    try:
        config = load_config()
        rec_cfg = config['recording']
        data_cfg = config['data']
        FRAME_WIDTH = rec_cfg['frame_width']
        FRAME_HEIGHT = rec_cfg['frame_height']
        DEFAULT_FPS = rec_cfg['default_fps']
        DATASET_DIR = data_cfg['dataset_dir'] # Resolved path from config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    sign_name = input("Enter sign name (e.g., 'hello'): ").strip().lower().replace(' ', '_')
    if not sign_name:
        print("Sign name cannot be empty.")
        return

    sign_dir = os.path.join(DATASET_DIR, sign_name)
    try:
        os.makedirs(sign_dir, exist_ok=True)
        print(f"Saving videos for '{sign_name}' in '{sign_dir}'")
    except OSError as e:
        print(f"Error creating directory {sign_dir}: {e}")
        return

    cap = cv2.VideoCapture(0) # Default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Warning: Camera FPS not detected, using default {DEFAULT_FPS}.")
        fps = DEFAULT_FPS

    recording = False
    frames_to_record = []

    print("\n--- Controls ---")
    print(" 's': Start/Resume recording")
    print(" 'q': Stop recording & Save current sample")
    print(" 'ESC': Exit (discard current recording if active)")
    print("----------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        display_frame = frame.copy()

        # Display status on frame
        status_text = "Press 's' to start"
        status_color = (0, 255, 0) # Green: Ready
        if recording:
            status_text = f"RECORDING... Frames: {len(frames_to_record)}. Press 'q' to save"
            status_color = (0, 0, 255) # Red: Recording
            # Blinking indicator
            if int(time.time() * 2) % 2 == 0:
                 cv2.circle(display_frame, (30, 30), 15, status_color, -1)
        else:
            cv2.circle(display_frame, (30, 30), 15, status_color, -1)

        cv2.putText(display_frame, status_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(f"Data Collection - Sign: {sign_name} | Press ESC to exit", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not recording:
            recording = True
            frames_to_record = [] # Start fresh
            print("\nStarted recording...")

        elif key == ord('q'): # Stop and Save
            if recording:
                recording = False
                print(f"Stopped. Recorded {len(frames_to_record)} frames.")

                if len(frames_to_record) > 10: # Save if reasonably long
                    video_id = str(uuid.uuid4())[:8]
                    video_filename = os.path.join(sign_dir, f"{sign_name}_{video_id}.mp4")
                    print(f"Saving to {video_filename} at {fps:.2f} FPS...")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = None
                    try:
                        out = cv2.VideoWriter(video_filename, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
                        if not out.isOpened(): raise IOError("VideoWriter failed.")
                        for recorded_frame in frames_to_record:
                            out.write(recorded_frame)
                        print("Video saved successfully.")
                    except Exception as e:
                        print(f"Error saving video: {e}")
                    finally:
                        if out: out.release()
                else:
                    print("Recording too short, not saved.")

                # Ask to continue for the same sign
                cont = input("Record another sample for this sign? (y/n): ").strip().lower()
                if cont != 'y': break
                else: print("\nReady for next sample. Press 's' to start.")
            else:
                 print("Not currently recording. Press 's' to start or ESC to exit.")

        elif key == 27: # ESC key
            if recording:
                print("\nESC pressed during recording. Discarding current sample.")
                recording = False
                frames_to_record = []
                # Ask to continue for the same sign even after discarding
                cont = input("Record another sample for this sign? (y/n): ").strip().lower()
                if cont != 'y': break
                else: print("\nReady for next sample. Press 's' to start.")
            else:
                print("\nExiting data collection.")
                break

        if recording:
            frames_to_record.append(frame) # Store original frame

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and resources released.")

if __name__ == "__main__":
    collect_data()