import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import tensorflow as tf
from fer import FER  # Facial Expression Recognition
from filterpy.kalman import KalmanFilter  # For smooth movement tracking
import math

class ISLParameterAnalyzer:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize models
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.face_detector = FER(mtcnn=True)
        
        # Initialize Kalman filter for movement tracking
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # x,y position and velocity
        self.initialize_kalman_filter()
        
        # Previous hand position for movement analysis
        self.prev_hand_position = None
        
    def initialize_kalman_filter(self):
        """Initialize Kalman filter parameters for movement tracking"""
        self.kf.F = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
        self.kf.R *= 0.1
        self.kf.Q *= 0.1
        
    def analyze_handshape(self, hand_landmarks):
        """Analyze hand shape using MediaPipe hand landmarks"""
        if not hand_landmarks:
            return None
            
        # Get finger states (extended/flexed)
        finger_states = []
        
        # Thumb
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_extended = thumb_tip.x > thumb_ip.x
        finger_states.append(thumb_extended)
        
        # Other fingers
        for finger in range(4):  # index, middle, ring, pinky
            tip_id = 8 + (finger * 4)
            pip_id = 6 + (finger * 4)
            
            finger_tip = hand_landmarks.landmark[tip_id]
            finger_pip = hand_landmarks.landmark[pip_id]
            
            # Check if finger is extended
            finger_extended = finger_tip.y < finger_pip.y
            finger_states.append(finger_extended)
            
        return {
            'thumb': finger_states[0],
            'index': finger_states[1],
            'middle': finger_states[2],
            'ring': finger_states[3],
            'pinky': finger_states[4]
        }
        
    def analyze_orientation(self, hand_landmarks):
        """Analyze hand orientation using palm normal vector"""
        if not hand_landmarks:
            return None
            
        # Calculate palm normal using cross product of palm vectors
        wrist = np.array([hand_landmarks.landmark[0].x,
                         hand_landmarks.landmark[0].y,
                         hand_landmarks.landmark[0].z])
        index_mcp = np.array([hand_landmarks.landmark[5].x,
                             hand_landmarks.landmark[5].y,
                             hand_landmarks.landmark[5].z])
        pinky_mcp = np.array([hand_landmarks.landmark[17].x,
                             hand_landmarks.landmark[17].y,
                             hand_landmarks.landmark[17].z])
        
        # Calculate vectors
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        # Calculate normal vector
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Calculate angles
        pitch = math.atan2(normal[1], normal[2])
        yaw = math.atan2(normal[0], normal[2])
        roll = math.atan2(v1[1], v1[0])
        
        return {
            'pitch': math.degrees(pitch),
            'yaw': math.degrees(yaw),
            'roll': math.degrees(roll)
        }
        
    def analyze_location(self, hand_landmarks, pose_landmarks):
        """Analyze hand location relative to body landmarks"""
        if not hand_landmarks or not pose_landmarks:
            return None
            
        # Get hand center
        hand_center = np.mean([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], axis=0)
        
        # Get key body landmarks
        nose = np.array([pose_landmarks.landmark[0].x,
                        pose_landmarks.landmark[0].y,
                        pose_landmarks.landmark[0].z])
        shoulder_mid = np.mean([
            [pose_landmarks.landmark[11].x, pose_landmarks.landmark[11].y, pose_landmarks.landmark[11].z],
            [pose_landmarks.landmark[12].x, pose_landmarks.landmark[12].y, pose_landmarks.landmark[12].z]
        ], axis=0)
        
        # Calculate relative positions
        relative_to_face = hand_center - nose
        relative_to_chest = hand_center - shoulder_mid
        
        return {
            'relative_to_face': relative_to_face.tolist(),
            'relative_to_chest': relative_to_chest.tolist(),
            'height_zone': self.get_height_zone(hand_center[1], pose_landmarks)
        }
        
    def analyze_movement(self, hand_landmarks):
        """Analyze hand movement using Kalman filter"""
        if not hand_landmarks:
            return None
            
        # Get hand center
        hand_center = np.mean([[lm.x, lm.y] for lm in hand_landmarks.landmark], axis=0)
        
        if self.prev_hand_position is None:
            self.prev_hand_position = hand_center
            return {'movement': 'static'}
            
        # Update Kalman filter
        self.kf.predict()
        self.kf.update(hand_center)
        
        # Calculate velocity
        velocity = self.kf.x[2:4]
        speed = np.linalg.norm(velocity)
        
        # Determine movement type
        movement_type = 'static'
        if speed > 0.1:
            angle = math.atan2(velocity[1], velocity[0])
            directions = ['right', 'up-right', 'up', 'up-left', 'left', 'down-left', 'down', 'down-right']
            direction_index = int((angle + math.pi) / (2 * math.pi / 8))
            movement_type = directions[direction_index]
            
        self.prev_hand_position = hand_center
        
        return {
            'movement': movement_type,
            'speed': speed
        }
        
    def analyze_non_manual(self, image):
        """Analyze facial expressions and non-manual features"""
        # Use FER for emotion detection
        emotions = self.face_detector.detect_emotions(image)
        
        if not emotions:
            return None
            
        # Get dominant emotion
        dominant_emotion = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'emotion_confidence': dominant_emotion[1],
            'all_emotions': emotions[0]['emotions']
        }
        
    def get_height_zone(self, y_coord, pose_landmarks):
        """Determine height zone of hand relative to body"""
        nose_y = pose_landmarks.landmark[0].y
        shoulder_y = pose_landmarks.landmark[11].y
        hip_y = pose_landmarks.landmark[23].y
        
        if y_coord < nose_y:
            return 'above_head'
        elif y_coord < shoulder_y:
            return 'head_level'
        elif y_coord < hip_y:
            return 'torso_level'
        else:
            return 'below_hip'
        
    def analyze_frame(self, frame):
        """Analyze all ISL parameters from a single frame"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.holistic.process(frame_rgb)
        
        # Analyze each parameter
        analysis = {
            'handshape': self.analyze_handshape(results.right_hand_landmarks),
            'orientation': self.analyze_orientation(results.right_hand_landmarks),
            'location': self.analyze_location(results.right_hand_landmarks, results.pose_landmarks),
            'movement': self.analyze_movement(results.right_hand_landmarks),
            'non_manual': self.analyze_non_manual(frame)
        }
        
        return analysis
