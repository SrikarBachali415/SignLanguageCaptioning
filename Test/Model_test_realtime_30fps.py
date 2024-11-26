import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import os
from datetime import datetime
import pandas as pd
import time

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants
MAX_SEQUENCE_LENGTH = 150
SLIDING_WINDOW_STEP = 30
FEATURE_DIM = 225
DESKTOP_PATH = os.path.expanduser("~/Desktop")
MODEL_PATH = r"Model\saved_model\saved_model.pb"

CLASSES = ['cold', 'cool', 'dry', 'happy', 'healthy', 'hot', 'loud', 'new', 'quiet', 'sick', 'warm']

class RealtimeSignLanguagePredictor:
    def __init__(self, save_predictions=True):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence_buffer = []
        self.frame_counter = 0
        self.save_predictions = save_predictions
        self.last_table_update = time.time()
        self.sequence_start_time = None
        self.csv_path = os.path.join(DESKTOP_PATH, f"sign_language_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
   
    def interpolate_missing_frames(self, landmarks):
        """Interpolate missing values in landmarks data"""
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
        """Extract features using the same method as training"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
        
        if pose.shape != (33, 3): pose = np.zeros((33, 3))
        if left_hand.shape != (21, 3): left_hand = np.zeros((21, 3))
        if right_hand.shape != (21, 3): right_hand = np.zeros((21, 3))
        
        features = np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
        return features, results

    def preprocess_sequence(self, sequence):
        """Preprocess sequence using the same method as training"""
        sequence = np.array(sequence)
        sequence = self.interpolate_missing_frames(sequence)
        
        # Always pad or trim to MAX_SEQUENCE_LENGTH
        if sequence.shape[0] < MAX_SEQUENCE_LENGTH:
            # For shorter sequences, pad with zeros
            padding = np.zeros((MAX_SEQUENCE_LENGTH - sequence.shape[0], FEATURE_DIM))
            sequence = np.vstack([sequence, padding])
        else:
            # For longer sequences, take the last MAX_SEQUENCE_LENGTH frames
            sequence = sequence[-MAX_SEQUENCE_LENGTH:]
            
        return sequence

    def make_predictions(self, frame_timestamp=None, sequence_start_time=None, force_predict=False):
        """Make predictions with option to force prediction on partial sequences"""
        if len(self.sequence_buffer) < MAX_SEQUENCE_LENGTH and not force_predict:
            return None

        # Preprocess sequence (will pad if needed)
        sequence = self.preprocess_sequence(self.sequence_buffer)
        sequence = np.expand_dims(sequence, axis=0)
        
        results = []
        for model_name, model in self.models.items():
            try:
                pred = model.signatures['serving_default'](tf.constant(sequence.astype(np.float32)))
                pred_array = next(iter(pred.values())).numpy()[0]
                
                top_3_indices = np.argsort(pred_array)[-3:][::-1]
                top_3_classes = [CLASSES[i] for i in top_3_indices]
                top_3_confidences = pred_array[top_3_indices]
                
                result_dict = {
                    'Model': model_name,
                    'Predicted_Class': top_3_classes[0],
                    'Confidence': top_3_confidences[0] * 100,
                    'Second_Guess': top_3_classes[1],
                    'Second_Confidence': top_3_confidences[1] * 100,
                    'Third_Guess': top_3_classes[2],
                    'Third_Confidence': top_3_confidences[2] * 100,
                    'Sequence_End_Time': frame_timestamp,
                    'Frame_Count': len(self.sequence_buffer),
                    'Is_Partial': len(self.sequence_buffer) < MAX_SEQUENCE_LENGTH
                }
                results.append(result_dict)
                
            except Exception as e:
                print(f"Error with model {model_name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if self.save_predictions:
            results_df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index=False)
            
        return results_df

    def process_remaining_buffer(self):
        """Process any remaining frames in the buffer"""
        if len(self.sequence_buffer) > 0:
            print(f"\nProcessing remaining {len(self.sequence_buffer)} frames...")
            final_predictions = self.make_predictions(
                frame_timestamp=time.time(),
                sequence_start_time=self.sequence_start_time,
                force_predict=True
            )
            if final_predictions is not None:
                print("\nFinal Predictions:")
                print("=" * 100)
                print(final_predictions.to_string(index=False))
            return final_predictions
        return None

    def draw_results(self, frame, results, landmarks_results):
        if landmarks_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks_results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        if landmarks_results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks_results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
        if landmarks_results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                landmarks_results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results is not None:
            y_pos = 30
            for _, row in results.iterrows():
                text = f"{row['Model']}: {row['Predicted_Class']} ({row['Confidence']:.1f}%)"
                if row.get('Is_Partial', False):
                    text += " (Partial)"
                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30

        buffer_status = f"Buffer: {len(self.sequence_buffer)}/{MAX_SEQUENCE_LENGTH}"
        cv2.putText(frame, buffer_status, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def print_prediction_table(self, predictions_df):
        if predictions_df is not None:
            print("\nLatest Predictions:")
            print("=" * 100)
            print(predictions_df.to_string(index=False))
            print(f"\nBuffer size: {len(self.sequence_buffer)}")
            print("\nPress 'q' to quit")

    def run(self, input_source=0):
        cap = cv2.VideoCapture(input_source)
        latest_predictions = None
        frames_since_last_prediction = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("\nReached end of video or no more frames available.")
                    break
                    
                if isinstance(input_source, str):
                    frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                else:
                    frame_time = time.time()
                
                features, landmarks_results = self.extract_features(frame)
                
                if len(self.sequence_buffer) == 0:
                    self.sequence_start_time = frame_time
                    
                self.sequence_buffer.append(features)
                frames_since_last_prediction += 1
                
                if len(self.sequence_buffer) >= MAX_SEQUENCE_LENGTH and frames_since_last_prediction >= SLIDING_WINDOW_STEP:
                    latest_predictions = self.make_predictions(
                        frame_timestamp=frame_time,
                        sequence_start_time=self.sequence_start_time
                    )
                    
                    self.sequence_buffer = self.sequence_buffer[SLIDING_WINDOW_STEP:]
                    frames_since_last_prediction = 0
                    self.sequence_start_time = frame_time

                self.draw_results(frame, latest_predictions, landmarks_results)
                cv2.imshow('Sign Language Detection', frame)
                
                current_time = time.time()
                if current_time - self.last_table_update >= 1.0 and latest_predictions is not None:
                    self.print_prediction_table(latest_predictions)
                    self.last_table_update = current_time
                
                wait_time = int(1000/fps) if isinstance(input_source, str) else 1
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    print("\nUser requested to quit.")
                    break

        finally:
            # Process any remaining frames before cleanup
            final_predictions = self.process_remaining_buffer()
            
            cap.release()
            cv2.destroyAllWindows()
            
            return final_predictions

def main():
    input_source = 0 
    predictor = RealtimeSignLanguagePredictor()
    print(f"Starting prediction using {'webcam' if isinstance(input_source, int) else 'video file'}. Press 'q' to quit.")
    final_predictions = predictor.run(input_source)

if __name__ == "__main__":
    main()