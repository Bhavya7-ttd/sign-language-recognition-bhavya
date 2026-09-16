# 🤟 Real-Time Sign Language Gesture Recognition System

A real-time computer vision and deep learning application that recognizes **American Sign Language (ASL)** hand gestures from a webcam feed and converts them into text, word suggestions, and spoken audio (Text-to-Speech).

---

## 📌 Overview

This project provides an end-to-end pipeline for recognizing ASL alphabets (A–Z) and control gestures in real time. It utilizes **MediaPipe** for 21 3D hand landmark extraction, standardizes coordinates onto a normalized skeleton canvas via a unified preprocessing pipeline, classifies gestures using a **8-group Convolutional Neural Network (CNN)** paired with landmark geometric rules, and features a responsive **Tkinter Graphical User Interface (GUI)** as well as a **Command Line Interface (CLI)**.

---

## ✨ Features

- 🎥 **Real-time Webcam Recognition**: High-FPS video capture with low-latency MediaPipe hand landmark detection.
- 📐 **Unified Canvas Preprocessing**: Left-hand mirroring, hand centering, and normalized 400x400 white skeleton canvas rendering for illumination/background invariance.
- 🧠 **Hybrid Classification Engine**: 8-super-group CNN model combined with precise 3D landmark geometric disambiguation rules.
- 🔤 **A–Z Alphabet & Control Gestures**: Supports all 26 ASL alphabet letters plus gesture-driven control actions:
  - 🤟 **Space**: Index & Pinky fingers extended UP (Middle & Ring folded down).
  - ✋ **Next**: Open palm with all fingers extended UP (commits staged character).
  - 👈 **Backspace**: Hand turned sideways with fingers pointing left.
- 🎛️ **Interactive Controls & FPS Counter**: Real-time FPS display, dynamic confidence percentage, and configurable stability threshold debouncing slider.
- 💡 **Word Suggestions & Text-to-Speech (TTS)**: Instant dictionary-based word completions via `pyenchant` and voice synthesis via `pyttsx3`.
- 🕹️ **Manual Camera Toggle**: Dedicated Start Cam ▶ and Stop Cam ⏹ buttons with safe model error handling.

---

## 🏗️ System Pipeline Architecture

```text
  Webcam Video Feed
         │
         ▼
MediaPipe Hand Detector (21 3D Landmarks)
         │
         ▼
Unified Canvas Preprocessing (Left-Hand Mirroring + Normalization + 400x400 Render)
         │
         ▼
CNN Super-Group Classifier + Geometric Landmark Rules
         │
         ▼
Debounced Character Selection & Staging
         │
         ▼
Word Suggestion Engine + Text Formation
         │
         ▼
Text-to-Speech (pyttsx3 Audio Output)
```

---

## 🧠 Technologies Used

| Technology | Role / Purpose |
| :--- | :--- |
| **Python 3.11** | Core application language |
| **OpenCV** | Image processing and real-time webcam frame acquisition |
| **MediaPipe** | 3D hand landmark detection and tracking |
| **TensorFlow / Keras** | CNN model loading, inference, and training |
| **NumPy** | Array transformations and geometric distance calculations |
| **Pillow (PIL)** | Image rendering for Tkinter GUI widgets |
| **pyttsx3** | Offline Text-to-Speech audio engine |
| **pyenchant** | Dictionary word search and auto-completion |
| **Tkinter** | Native dark-themed Desktop GUI framework |

---

## 📂 Project Structure

```text
Sign-Language-Recognition/
│
├── AtoZ_3.1/                       # Dataset directory (A-Z subfolders)
│
├── Gesture detection/               # Gesture detection module and helper assets
│
├── cnn8grps_rad1_model.h5          # Trained 8-group CNN model weights
├── hand_landmarker.task             # MediaPipe hand landmarker task file
│
├── collect_data_binary_images.py   # Dataset image collection utility
├── collect_data_skeleton.py        # Skeleton canvas collection utility
├── sign_language_cli.py            # Command Line Interface (CLI) recognition runner
├── sign_language_gui.py            # Main Interactive Graphical User Interface (GUI)
├── test_hand_tracking.py           # Quick MediaPipe webcam diagnostic test script
│
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
└── demo.mp4                        # Video demonstration recording
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10 or 3.11** installed on your system.

### 2. Setup Virtual Environment & Install Dependencies

```powershell
# Clone the repository
git clone https://github.com/Bhavya7-ttd/sign-language-recognition-bhavya.git
cd sign-language-recognition-bhavya

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Graphical User Interface (GUI)
Launch the primary interactive desktop interface:
```powershell
python sign_language_gui.py
```
* **Start Cam ▶**: Starts the webcam feed and gesture prediction loop.
* **Stop Cam ⏹**: Pauses the camera feed.
* **Stability Slider**: Adjusts the number of consecutive identical frames required before confirming a gesture (default: 5 frames).
* **Confidence Display**: Shows live prediction probability percentage.

### Command Line Interface (CLI)
Run gesture recognition directly in your terminal:
```powershell
python sign_language_cli.py
```

### Model Evaluation
To evaluate model performance on the `AtoZ_3.1` dataset:
```powershell
python evaluate.py
```

---

## 🤟 Gesture Control Reference

| Action | Physical Hand Gesture | GUI Button Alternative |
| :--- | :--- | :--- |
| **Space (`" "`)** | Index & Pinky fingers UP (Middle & Ring folded down) | Click **`Space ␣`** |
| **Next / Commit (`"next"`)** | Open Palm (All 5 fingers extended UP) | Click **`Next`** |
| **Backspace (`"Backspace"`)** | Hand turned sideways (Fingers pointing left) | Click **`Backspace ⌫`** |

---

## 📜 License

This project is open-source and available under the **MIT License**.
