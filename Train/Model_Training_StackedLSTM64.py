import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Add dataset path below
CSV_PATH = "Adjectives_videos.csv"
MODEL_SAVE_PATH = r"Model\saved_model\saved_model.pb"
MAX_SEQUENCE_LENGTH = 150  # Fixed frame count per video
FEATURE_DIM = 225  # Pose (33), left hand (21), right hand (21), all with x, y, z

# Mediapipe Holistic
mp_holistic = mp.solutions.holistic

# Helper Functions
def read_video(video_path):
    """Reads frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def interpolate_missing_frames(landmarks):
    """
    Interpolates missing landmarks between frames.
    Fills the first and last missing frames with zeros.
    """
    for i in range(landmarks.shape[1]):
        column = landmarks[:, i]
        nan_indices = np.isnan(column)
        if np.any(nan_indices):
            not_nan = np.where(~nan_indices)[0]
            if not_nan.size > 0:
                column[nan_indices] = np.interp(np.where(nan_indices)[0], not_nan, column[not_nan])
            else:
                column[:] = 0  # All missing
        landmarks[:, i] = column
    return landmarks

def extract_features(video_path):
    """Extract skeletal key points from video frames using MediaPipe."""
    holistic = mp_holistic.Holistic(static_image_mode=False)
    frames = read_video(video_path)
    video_data = []

    for frame in frames:
        results = holistic.process(frame)

        # Extract pose landmarks
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
        # Left hand landmarks
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
        # Right hand landmarks
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))

        # Ensure the correct dimensions for each group
        if pose.shape != (33, 3):
            pose = np.zeros((33, 3))  # Fallback to zeros if incorrect
        if left_hand.shape != (21, 3):
            left_hand = np.zeros((21, 3))
        if right_hand.shape != (21, 3):
            right_hand = np.zeros((21, 3))

        # Combine all landmarks into a single frame dictionary
        frame_data = {"pose": pose, "left_hand": left_hand, "right_hand": right_hand}
        video_data.append(frame_data)

    holistic.close()

    # Interpolate missing frames
    for i, frame_data in enumerate(video_data):
        video_data[i] = interpolate_missing_frames(frame_data)

    # Normalize key points
    normalized_video_data = []
    for frame_data in video_data:
        pose = frame_data["pose"]
        left_hand = frame_data["left_hand"]
        right_hand = frame_data["right_hand"]

        # Calculate the center based on pose landmarks 11 and 12
        center = (pose[11] + pose[12]) / 2 if np.any(pose[11]) and np.any(pose[12]) else np.zeros(3)

        # Translate all landmarks to make the center the origin
        pose -= center
        left_hand -= center
        right_hand -= center

        # Normalize to range [-1, 1] based on the max absolute value per frame
        max_value = np.max(np.abs(np.concatenate([pose, left_hand, right_hand])))
        if max_value > 0:  # Avoid division by zero
            pose /= max_value
            left_hand /= max_value
            right_hand /= max_value

        # Store the normalized landmarks in the same structure
        normalized_video_data.append({
            "pose": pose,
            "left_hand": left_hand,
            "right_hand": right_hand
        })

    return normalized_video_data

def preprocess_data(csv_path, quick_train=False):
    """Processes dataset, filtering classes with fewer than 18 videos."""
    df = pd.read_csv(csv_path)

    # Optional: Quick training with selected classes
    if quick_train:
        quick_classes = ['loud', 'quiet', 'sick', 'healthy']
        df = df[df['Adjectives'].isin(quick_classes)]

    # Filter classes with fewer than 18 videos
    class_counts = df['Adjectives'].value_counts()
    valid_classes = class_counts[class_counts >= 18].index
    df = df[df['Adjectives'].isin(valid_classes)]

    print(f"Classes retained after filtering: {valid_classes.tolist()}")
    print(f"Total samples after filtering: {len(df)}")

    labels = df['Adjectives'].values
    paths = df['Path'].values

    features = []
    for path in paths:
        print(f"Processing: {path}")
        video_features = extract_features(path)

        # Pad or truncate sequence to fixed length
        if video_features.shape[0] < MAX_SEQUENCE_LENGTH:
            padding = np.zeros((MAX_SEQUENCE_LENGTH - video_features.shape[0], FEATURE_DIM))
            video_features = np.vstack([video_features, padding])
        elif video_features.shape[0] > MAX_SEQUENCE_LENGTH:
            video_features = video_features[:MAX_SEQUENCE_LENGTH]

        features.append(video_features)

    features = np.array(features)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    labels_onehot = onehot_encoder.fit_transform(labels_encoded.reshape(-1, 1))

    return features, labels_onehot, label_encoder.classes_


# Build Model
def build_lstm_model(input_shape, num_classes):
    """Defines the LSTM model architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh'),
        tf.keras.layers.LSTM(64, return_sequences=False, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot Training Metrics
def plot_metrics(history):
    """Plots training and validation accuracy and loss."""
    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(['accuracy', 'loss']):
        plt.subplot(1, 2, i + 1)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(metric.capitalize())
        plt.legend()
    plt.show()

# Train and Evaluate
def train_and_evaluate(quick_train=False):
    """Trains and evaluates the LSTM model."""
    features, labels, classes = preprocess_data(CSV_PATH, quick_train=quick_train)

    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")

    model = build_lstm_model(input_shape=(MAX_SEQUENCE_LENGTH, FEATURE_DIM), num_classes=labels.shape[1])
    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32)

    # Save model
    model.save(MODEL_SAVE_PATH)

    # Evaluate model
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    y_pred = model.predict(x_val).argmax(axis=1)
    y_true = y_val.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=classes).plot()
    plt.show()

    # Plot Metrics
    plot_metrics(history)

# Main Execution
if __name__ == "__main__":
    train_and_evaluate(quick_train=False)
