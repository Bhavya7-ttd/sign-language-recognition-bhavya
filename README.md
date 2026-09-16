# 🤟 Real-Time Sign Language Gesture Recognition System

A real-time computer vision and deep learning desktop application that recognizes **American Sign Language (ASL)** hand gestures from a webcam feed and converts them into text, word suggestions, and spoken audio using Text-to-Speech.

---

## 📌 Overview

This project provides an end-to-end pipeline for recognizing ASL alphabets (A–Z) and control gestures in real time. The system utilizes **MediaPipe** for 21 3D hand landmark extraction, standardizes hand coordinates onto a normalized skeleton canvas through a unified preprocessing pipeline, and classifies gestures using an **8-group Convolutional Neural Network (CNN)** combined with landmark-based geometric rules. The application provides both a responsive **Tkinter Graphical User Interface (GUI)** and a **Command Line Interface (CLI)** for gesture recognition.

---

## ✨ Features

- 🎥 **Real-time Webcam Recognition**
  - Captures live webcam frames and performs real-time hand gesture recognition.
- 📐 **Unified Canvas Preprocessing**
  - Detects hand landmarks.
  - Centers and normalizes the hand.
  - Supports left-hand mirroring.
  - Renders the hand skeleton on a standardized **400 × 400** canvas.
- 🧠 **Hybrid Classification Engine**
  - Uses an **8-super-group CNN model** for gesture classification.
  - Uses landmark-based geometric rules to distinguish visually similar gestures.
- 🔤 **A–Z Alphabet Recognition**
  - Supports recognition of all 26 ASL alphabet letters.
- 📝 **Gesture-to-Text Conversion**
  - Converts recognized gestures into characters and forms words and sentences.
- 💡 **Word Suggestions**
  - Provides dictionary-based word suggestions using `pyenchant`.
- 🔊 **Text-to-Speech**
  - Converts generated text into speech using `pyttsx3`.
- 🎛️ **Confidence Display**
  - Displays the model's prediction confidence in real time.
- ⚡ **Gesture Stability / Debouncing**
  - Uses consecutive-frame confirmation to reduce accidental predictions.
- 🖥️ **Interactive GUI**
  - Provides webcam controls, prediction display, text formation, word suggestions, and other controls.
- 🕹️ **Manual Camera Controls**
  - Includes dedicated **Start Cam ▶** and **Stop Cam ⏹** controls.
- 💻 **Command Line Interface**
  - Supports gesture recognition directly from the terminal.

---

## 🏗️ System Pipeline

```text
Webcam Video Feed
       │
       ▼
MediaPipe Hand Detector (21 3D Hand Landmarks)
       │
       ▼
Unified Canvas Preprocessing (Left-Hand Mirroring + Normalization + 400 × 400 Skeleton Canvas)
       │
       ▼
CNN Super-Group Classifier
       │
       ▼
Geometric Landmark Rules
       │
       ▼
Debounced Character Selection
       │
       ▼
Word Suggestion Engine
       │
       ▼
Text Formation
       │
       ▼
Text-to-Speech (pyttsx3)
```

---

## 🧠 Technologies

| Technology | Role / Purpose |
| :--- | :--- |
| **Python 3.11** | Core application language |
| **OpenCV** | Image processing and real-time webcam frame acquisition |
| **MediaPipe** | 3D hand landmark detection and tracking |
| **TensorFlow / Keras** | CNN model loading, inference, and training |
| **NumPy** | Array transformations and geometric calculations |
| **Pillow (PIL)** | Image rendering for Tkinter GUI widgets |
| **pyttsx3** | Offline Text-to-Speech audio engine |
| **pyenchant** | Dictionary word search and auto-completion |
| **Tkinter** | Desktop graphical user interface |

---

## 📂 Project Structure

```text
Sign-Language-Recognition/
│
├── AtoZ_3.1/
│   └── A-Z dataset folders
│
├── Gesture detection/
│   └── Gesture detection module and helper assets
│
├── cnn8grps_rad1_model.h5
│   └── Trained 8-group CNN model
│
├── hand_landmarker.task
│   └── MediaPipe hand landmarker model
│
├── preprocessing.py
│   └── Unified landmark and canvas preprocessing
│
├── sign_language_gui.py
│   └── Main desktop GUI application
│
├── sign_language_cli.py
│   └── Command Line Interface recognition runner
│
├── collect_data_binary_images.py
│   └── Binary image dataset collection utility
│
├── collect_data_skeleton.py
│   └── Skeleton canvas dataset collection utility
│
├── train.py
│   └── CNN model training script
│
├── evaluate.py
│   └── Model evaluation script
│
├── test_hand_tracking.py
│   └── MediaPipe webcam diagnostic test
│
├── requirements.txt
│   └── Project dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Git ignore rules
│
└── demo.mp4
    └── Project demonstration video
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Make sure you have **Python 3.10** or **Python 3.11** installed.
* The project is designed to run on Windows with a webcam.

### 2. Clone the Repository
```powershell
git clone https://github.com/Bhavya7-ttd/sign-language-recognition-bhavya.git
cd sign-language-recognition-bhavya
```

### 3. Create a Virtual Environment
```powershell
python -m venv venv
```

### 4. Activate the Virtual Environment
On Windows PowerShell:
```powershell
.\venv\Scripts\activate
```

### 5. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Graphical User Interface (GUI)
Launch the main application:
```powershell
python sign_language_gui.py
```

The GUI provides:
* **Start Cam ▶** — Starts webcam recognition.
* **Stop Cam ⏹** — Stops the webcam feed.
* **Stability Slider** — Controls the number of consecutive frames required to confirm a gesture.
* **Confidence Display** — Shows the prediction confidence.
* **Word Suggestions** — Displays possible words based on recognized characters.
* **Text Formation** — Builds words and sentences from recognized gestures.
* **Text-to-Speech** — Converts generated text into spoken audio.

### Command Line Interface (CLI)
Run:
```powershell
python sign_language_cli.py
```
The CLI allows gesture recognition directly from the terminal.

---

## 🤟 Gesture Controls

The application supports additional gestures for text editing and sentence formation.

| Action | Physical Hand Gesture | GUI Button Alternative |
| :--- | :--- | :--- |
| **Space (`" "`)** | Index & Pinky fingers UP; Middle & Ring fingers folded | Click **`Space ␣`** |
| **Next / Commit (`"next"`)** | Open palm with all five fingers extended UP | Click **`Next`** |
| **Backspace** | Hand turned sideways with fingers pointing left | Click **`Backspace ⌫`** |

> **Note:** Gesture descriptions correspond to the gestures implemented in the current application code.

---

## 🧠 CNN Classification Approach

Instead of directly classifying all 26 alphabet letters using a single output class for every letter, the system uses an **8-group CNN classification approach**.

The CNN first predicts a super-group containing visually related gestures. Landmark-based geometric rules are then used to distinguish individual gestures within the predicted group.

```text
Hand Gesture
     │
     ▼
MediaPipe Landmarks
     │
     ▼
Skeleton Canvas
     │
     ▼
8-Group CNN
     │
     ▼
Super-Group
     │
     ▼
Geometric Landmark Rules
     │
     ▼
Final A–Z Gesture
```

This hybrid approach combines image-based CNN classification with hand landmark geometry.

---

## 📐 Preprocessing

The preprocessing pipeline is implemented in:
`preprocessing.py`

The preprocessing stage standardizes the hand representation before classification. The pipeline includes:
* Hand landmark extraction
* Coordinate normalization
* Hand centering
* Left-hand mirroring
* Skeleton rendering
* Standardized **400 × 400** canvas generation

Using a standardized representation helps reduce the effect of different hand positions and backgrounds on the classification input.

---

## 📊 Model Evaluation

The model was evaluated using a stratified 20% held-out dataset split containing **935 images** across the 26 ASL alphabet categories.

### Held-Out Test Result
* **Correct Predictions**: 935 / 935
* **Accuracy**: 100%

> **Important:** This result represents performance on the evaluated held-out dataset split. It should not be interpreted as 100% real-world accuracy across different users, environments, cameras, lighting conditions, or backgrounds.

---

## 🔬 Recognition Process

For each webcam frame, the system performs the following steps:
1. Captures the webcam frame.
2. Detects the hand using MediaPipe.
3. Extracts 21 hand landmarks.
4. Processes and normalizes the landmark coordinates.
5. Generates a standardized **400 × 400** skeleton canvas.
6. Passes the processed image to the CNN model.
7. Predicts one of the CNN's super-groups.
8. Applies geometric landmark rules.
9. Determines the final alphabet character or control gesture.
10. Applies stability/debouncing.
11. Updates the text buffer.
12. Provides word suggestions when applicable.
13. Converts the final text into speech when requested.

---

## 💡 Word Suggestions

The application provides dictionary-based word suggestions using `pyenchant`.

For example:
* **Recognized characters**: `HEL`
* **Possible suggestions**: `HELP`, `HELLO`, `HELMET`

The user can select an available suggestion to complete the current word.

---

## 🔊 Text-to-Speech

The project uses `pyttsx3` for offline Text-to-Speech functionality. The recognized text can be converted into spoken audio through the application's TTS functionality.

---

## 🧪 Testing

The application was tested for functionality including:
* A–Z gesture recognition
* Webcam hand detection
* Hand landmark tracking
* Left-hand input & Right-hand input
* Different hand positions & distances
* Lighting variations
* Confidence display & Gesture stability
* Word formation & Sentence formation
* Space control, Backspace control, Next / commit control
* Word suggestions & Text-to-Speech
* Camera Start / Stop controls
* Model loading and error handling

---

## 📈 Performance

* The CNN inference stage runs in approximately **5–15 ms** per frame in testing.
* The GUI also provides a real-time FPS display.
* Actual performance may vary depending on computer hardware, webcam, lighting, and runtime conditions.

---

## 🎥 Demo

A demonstration video of the application is included in the repository:
`demo.mp4`

The demo shows the real-time webcam recognition system and its main GUI functionality.

---

## ⚠️ Limitations

* Recognition performance depends on webcam quality and environmental conditions.
* The system is primarily designed around the supported A–Z gesture dataset.
* Real-world performance may vary between users.
* Hand position, camera distance, lighting, and background can affect recognition.
* Dataset accuracy does not necessarily represent real-world recognition accuracy.
* The current system focuses primarily on alphabet-level gesture recognition and application-level text formation.
* Dynamic sign-language gestures and complete continuous sign-language sentences are not the primary focus of the current model.

---

## 🔮 Future Scope

Possible future improvements include:
* Recognition of dynamic sign-language gestures
* Recognition of complete words and phrases
* Larger and more diverse datasets
* Improved real-world generalization
* Continuous sign-language sentence recognition
* AI-assisted sentence correction
* Natural Language Processing (NLP)
* Multilingual Text-to-Speech
* Mobile application support
* Cloud-based deployment
* Improved gesture personalization
* Support for additional sign languages

---

## 👩‍💻 Author

**Bhavya Ramisetty**  
GitHub: [https://github.com/Bhavya7-ttd](https://github.com/Bhavya7-ttd)

---

## 📜 License

This project is open-source and available under the **MIT License**.
