<img width="456" height="602" alt="Screenshot (1)" src="https://github.com/user-attachments/assets/9a5b34fb-a799-44c8-8789-73786933b61a" />



# 😊 Real-Time Facial Emotion Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A real-time facial emotion recognition system using a deep CNN model and OpenCV face detection.**

[📌 Features](#-features) · [🧠 Model Architecture](#-model-architecture) · [🚀 Getting Started](#-getting-started) · [📁 Project Structure](#-project-structure) · [📊 Emotion Classes](#-emotion-classes)

</div>

---

## 📌 Features

- 🎥 **Real-time detection** — processes live webcam video frame by frame
- 🧠 **Deep CNN model** — custom Xception-inspired architecture trained on facial expression data
- 👤 **Face detection** — uses OpenCV's Haar Cascade classifier to isolate faces from frames
- 😄 **7 Emotion classes** — Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- ⚡ **Lightweight inference** — grayscale 64×64 input keeps prediction fast
- 🔖 **On-frame label overlay** — predicted emotion rendered directly on the video feed

---

## 🧠 Model Architecture

The model (`emotion_model.hdf5`) is a custom **Xception-inspired CNN** built with Keras/TensorFlow. It uses depthwise separable convolutions with residual (skip) connections — making it efficient and accurate.

| Layer Block | Details |
|---|---|
| **Input** | `(64, 64, 1)` — grayscale face image |
| **Entry Flow** | 2× Conv2D (8 filters, 3×3) + BatchNorm + ReLU |
| **Residual Block 1** | 2× SeparableConv2D (16 filters) + MaxPool + Skip Conv2D |
| **Residual Block 2** | 2× SeparableConv2D (32 filters) + MaxPool + Skip Conv2D |
| **Residual Block 3** | 2× SeparableConv2D (64 filters) + MaxPool + Skip Conv2D |
| **Residual Block 4** | 2× SeparableConv2D (128 filters) + MaxPool + Skip Conv2D |
| **Exit Flow** | Conv2D (7 filters, 3×3) + GlobalAveragePooling2D |
| **Output** | Softmax → 7 emotion classes |

**Key design choices:**
- `SeparableConv2D` for parameter efficiency
- `BatchNormalization` (momentum=0.99) after every conv layer for training stability
- `L2 regularization` (λ=0.01) on entry conv layers to reduce overfitting
- `Add` (residual skip connections) between blocks for gradient flow
- `GlobalAveragePooling2D` instead of Flatten — avoids overfitting on spatial features

---

## 📊 Emotion Classes

The model predicts one of **7 universal facial expressions**:

| Index | Emotion | Description |
|---|---|---|
| 0 | 😠 Angry | Frustration, hostility |
| 1 | 🤢 Disgust | Aversion, revulsion |
| 2 | 😨 Fear | Anxiety, terror |
| 3 | 😄 Happy | Joy, pleasure |
| 4 | 😢 Sad | Sorrow, disappointment |
| 5 | 😲 Surprise | Astonishment, shock |
| 6 | 😐 Neutral | No strong expression |

---

## 📁 Project Structure

```
emotion-detection/
├── emotion_detection.py                 # Main script — webcam capture + inference loop
├── emotion_model.hdf5                   # Pre-trained Keras model (Xception-style CNN)
├── haarcascade_frontalface_default.xml  # OpenCV Haar Cascade for face detection
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- A working webcam

### 1. Clone the Repository

```bash
git clone https://github.com/dipikadas6979/emotion-detection.git
cd emotion-detection
```

### 2. Install Dependencies

```bash
pip install tensorflow==2.x opencv-python numpy
```

> **Note:** The model was trained with **Keras 2.0.5** / **TensorFlow 2.x**. For best compatibility, use TensorFlow 2.0–2.9.

### 3. Run the Application

```bash
python emotion_detection.py
```

A webcam window will open. Detected faces will be boxed and labelled with the predicted emotion in real time. Press **`q`** to quit.

---

## ⚙️ How It Works

```
Webcam Frame
    │
    ▼
Convert to Grayscale
    │
    ▼
Haar Cascade Face Detection  ◄── haarcascade_frontalface_default.xml
    │
    ▼
Crop & Resize Face → (64, 64, 1)
    │
    ▼
Normalize Pixel Values (÷ 255)
    │
    ▼
CNN Inference  ◄── emotion_model.hdf5
    │
    ▼
Softmax → 7 Probabilities
    │
    ▼
argmax → Predicted Emotion Label
    │
    ▼
Overlay Label on Frame → Display
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Deep Learning Framework | TensorFlow / Keras 2.0.5 |
| Computer Vision | OpenCV (`cv2`) |
| Numerical Computing | NumPy |
| Model Format | HDF5 (`.hdf5`) |
| Face Detection | Haar Cascade (XML) |

---

## 📦 Dependencies

```txt
tensorflow>=2.0
opencv-python>=4.0
numpy>=1.18
```

Install all at once:

```bash
pip install tensorflow opencv-python numpy
```

---

## 🔧 Customization

**Use a static image instead of webcam:**
```python
frame = cv2.imread('your_image.jpg')
```

**Save output video:**
```python
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (640, 480))
out.write(frame)
```

**Adjust face detection sensitivity:**
```python
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```

---

## ⚠️ Notes & Limitations

- Performance may vary with poor lighting or extreme head angles
- The Haar Cascade detector works best on frontal, unobstructed faces
- For production use, consider upgrading to a DNN-based face detector (e.g., `cv2.dnn`)
- The model is trained for single-face detection; multi-face scenarios depend on implementation

---

## 🙏 Acknowledgements

- Model architecture inspired by **Xception** (Chollet, 2017)
- Face detection via **OpenCV Haar Cascades**
- Emotion labels based on **Ekman's Universal Emotions** (+ Neutral)
- Mentored by **SK Sahil** — AI Developer & Tutor at [Code_ScholarEU](https://www.instagram.com/code_scholar_eu/)

---

## 👩‍💻 Author

**Dipika Das**
B.Sc (Computer Science) · Haldia Institute of Management

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/dipikadas6979)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dipika-das-83895a397/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white)](https://www.instagram.com/dip_ika9857)

---

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!**

</div>
