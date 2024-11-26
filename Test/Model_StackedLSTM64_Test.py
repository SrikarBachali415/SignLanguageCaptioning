import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import mediapipe as mp
import cv2
from collections import deque
from sklearn.preprocessing import LabelEncoder

# Constants
MAX_SEQUENCE_LENGTH = 150
FEATURE_DIM = 225
MODEL_PATH = r"\adjectives_lstm_model_15_2.keras"
OUTPUT_PATH = os.path.join(os.path.expanduser("~"), "Desktop", "isl_prediction_output_v15.avi")

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class RealTimePredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print(self.model.summary())
        # Update class names based on your v15 model's classes
        self.class_names = ['loud', 'hot', 'healthy', 'cool', 'sick', 'warm', 'dry', 'quiet', 'new', 'happy', 'cold']
        self.frame_buffer = deque(maxlen=MAX_SEQUENCE_LENGTH)
        self.holistic = mp_holistic.Holistic(static_image_mode=False)
        self.is_paused = False
        self.playback_speed = 1.0

    def interpolate_missing_frames(self, landmarks):
        """Interpolates missing landmarks between frames."""
        for i in range(landmarks.shape[1]):
            column = landmarks[:, i]
            nan_indices = np.isnan(column)
            if np.any(nan_indices):
                not_nan = np.where(~nan_indices)[0]
                if not_nan.size > 0:
                    column[nan_indices] = np.interp(np.where(nan_indices)[0], not_nan, column[not_nan])
                else:
                    column[:] = 0
            landmarks[:, i] = column
        return landmarks

    def extract_features(self, frame):
        """Extract features from a single frame using the same method as training."""
        results = self.holistic.process(frame)
        
        # Extract pose landmarks
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        # Left hand landmarks
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        # Right hand landmarks
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

        # Ensure correct dimensions
        if pose.shape != (33, 3):
            pose = np.zeros((33, 3))
        if left_hand.shape != (21, 3):
            left_hand = np.zeros((21, 3))
        if right_hand.shape != (21, 3):
            right_hand = np.zeros((21, 3))

        # Combine features
        features = np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
        
        return features, results

    def get_top_5_predictions(self, features):
        """Get top 5 predictions from the model."""
        if len(self.frame_buffer) < MAX_SEQUENCE_LENGTH:
            return [], []
        
        # Convert buffer to numpy array
        sequence = np.array(list(self.frame_buffer))
        
        # Interpolate missing frames
        sequence = self.interpolate_missing_frames(sequence)
        
        # Prepare for prediction
        sequence = np.expand_dims(sequence, axis=0)
        
        predictions = self.model.predict(sequence, verbose=0)[0]
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        
        top_5_classes = [self.class_names[i] for i in top_5_indices]
        top_5_confidences = predictions[top_5_indices]
        
        # Print high confidence predictions
        if top_5_confidences[0] >= 0.90:
            print(f"\nHigh confidence detection!")
            print(f"Sign: {top_5_classes[0]}")
            print(f"Confidence: {top_5_confidences[0]:.1%}")
        
        return top_5_classes, top_5_confidences

    def draw_predictions(self, frame, top_5_classes, top_5_confidences):
        """Draw predictions and controls on the frame."""
        # Background for predictions
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 220), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw predictions
        for i, (cls, conf) in enumerate(zip(top_5_classes, top_5_confidences)):
            y_pos = 40 + (i * 25)
            text = f"{i+1}. {cls}: {conf:.1%}"
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw buffer status
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{MAX_SEQUENCE_LENGTH}"
        cv2.putText(frame, buffer_text, (20, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw playback controls
        status = "PAUSED" if self.is_paused else "PLAYING"
        controls_text = f"Status: {status} | Speed: {self.playback_speed}x"
        cv2.putText(frame, controls_text, (20, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw control instructions
        instructions = "Controls: SPACE=Pause | ←→=Speed | Q=Quit"
        cv2.putText(frame, instructions, (frame.shape[1] - 400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def process_video(self, video_path):
        """Process video with real-time predictions and playback controls."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        
        top_5_classes, top_5_confidences = [], []
        frame_time = int(1000 / fps)
        
        while cap.isOpened():
            if not self.is_paused:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                features, results = self.extract_features(rgb_frame)
                self.frame_buffer.append(features)
                
                if len(self.frame_buffer) >= MAX_SEQUENCE_LENGTH:
                    top_5_classes, top_5_confidences = self.get_top_5_predictions(features)
                
                # Draw landmarks and predictions
                annotated_frame = draw_landmarks_on_frame(rgb_frame, results)
                annotated_frame = self.draw_predictions(annotated_frame, top_5_classes, top_5_confidences)
                
                out.write(annotated_frame)
            else:
                # When paused, keep showing the last frame
                annotated_frame = self.draw_predictions(annotated_frame.copy(), top_5_classes, top_5_confidences)
            
            cv2.imshow('Real-time ISL Recognition (v15)', annotated_frame)
            
            # Handle keyboard input
            wait_time = int(frame_time / self.playback_speed)
            key = cv2.waitKey(wait_time if not self.is_paused else 1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                self.is_paused = not self.is_paused
                print(f"\nPlayback {'paused' if self.is_paused else 'resumed'}")
            elif key == 83:  # Right arrow
                self.playback_speed = min(4.0, self.playback_speed + 0.25)
                print(f"\nPlayback speed: {self.playback_speed}x")
            elif key == 81:  # Left arrow
                self.playback_speed = max(0.25, self.playback_speed - 0.25)
                print(f"\nPlayback speed: {self.playback_speed}x")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.holistic.close()
        print(f"\nOutput video saved as: {OUTPUT_PATH}")

def draw_landmarks_on_frame(frame, results):
    """Draw the landmarks on the frame."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
    
    return frame

def main():
    predictor = RealTimePredictor()
    video_path = r"test_video.mp4"
    predictor.process_video(video_path)

if __name__ == "__main__":
    main()