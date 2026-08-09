# 🤟 Sign Language Recognition

A real-time computer vision application that recognizes American Sign Language (ASL) hand gestures using a webcam and converts the recognized gestures into text and speech.

## 📌 Overview

This project uses computer vision and machine learning techniques to recognize hand gestures representing American Sign Language alphabets.

The system captures hand movements through a webcam, detects hand landmarks using MediaPipe, processes the landmark information, predicts the corresponding gesture, and displays the recognized character. The recognized text can also be converted into speech.

## ✨ Features

- 🎥 Real-time webcam-based hand gesture recognition
- ✋ Hand landmark detection using MediaPipe
- 🔤 American Sign Language alphabet recognition
- 📝 Gesture-to-text conversion
- 🔊 Text-to-speech output
- ⚡ Real-time prediction
- 🖥️ Interactive graphical interface
- 🌐 Works with different backgrounds and lighting conditions using hand landmarks

## 🏗️ System Architecture

The overall pipeline is:

Webcam
↓
Hand Detection
↓
MediaPipe Hand Landmarks
↓
Landmark Processing
↓
Gesture Classification
↓
Recognized Character
↓
Text Formation
↓
Text-to-Speech

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Computer vision and webcam processing |
| MediaPipe | Hand detection and landmark extraction |
| TensorFlow / Keras | Machine learning and gesture classification |
| NumPy | Numerical processing |
| pyttsx3 | Text-to-speech conversion |

## 📂 Project Structure

```text
Sign-Language-Recognition/
│
├── AtoZ_3.1/
│
├── Gesture detection/
│
├── cnn8grps_rad1_model.h5
├── hand_landmarker.task
│
├── collect_data_binary_images.py
├── collect_data_skeleton.py
├── sign_language_cli.py
├── sign_language_gui.py
├── test_hand_tracking.py
│
├── README.md
├── .gitignore
└── demo.mp4
