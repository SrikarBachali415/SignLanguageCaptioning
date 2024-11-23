# Indian Sign Language (ISL) Captioning

A deep learning-based project that translates live video of Indian Sign Language (ISL) gestures into real-time captions in English. This system utilizes **MediaPipe** for hand detection and **LSTM** for sequence modeling.

## Features
- **Real-Time Processing**: Captures live video input and generates English captions instantly.
- **Gesture Recognition**: Utilizes MediaPipe Hands for detecting hand keypoints.
- **Sequence Modeling**: Employs Long Short-Term Memory (LSTM) networks for gesture-to-text translation.
- **Scalability**: Can be extended to support additional gestures and vocabulary.

## Technologies Used
- **Hand Detection**: MediaPipe Holistic
- **Sequence Modeling**: LSTM
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Video Processing**: OpenCV

## Prerequisites
- Python 3.7 or later
- A webcam for video input
- A virtual environment (recommended)
- Libraries specified in `requirements.txt`

## Setup Instructions

### Installation
1. Clone the repository:
   git clone https://github.com/yourusername/isl-captioning.git
   cd isl-captioning
2. Ensure your webcam is connected. The system will capture ISL gestures and display captions in real-time.

## Project Structure
```plaintext
├── mediapipe_models/   # Configurations for hand tracking
├── lstm_model/         # Pre-trained weights and LSTM model
├── data/               # Training datasets and annotations
├── scripts/            # Helper scripts for training and preprocessing
├── main.py             # Entry point for running the application
├── requirements.txt    # Python dependencies
```
## Flow Diagram
![flowv2 drawio](https://github.com/user-attachments/assets/8658480b-47d4-46e6-a979-bf48b64a472f)


## Future Enhancements
-Vocabulary Expansion: Extend the system to recognize more gestures and phrases.
-Graphical User Interface (GUI): Implement a user-friendly interface for non-technical users.
-Optimization: Improve the real-time performance for deployment on low-resource devices.
-Mobile App Integration: Build a mobile application to make the system more portable.

## Usage
-Perform ISL gestures in front of the webcam.
-The system processes the gestures and displays live captions in English.

## Challenges and Limitations
-Requires consistent lighting for accurate hand tracking.
-Currently supports a limited vocabulary.