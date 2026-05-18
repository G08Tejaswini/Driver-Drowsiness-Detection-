# Driver Drowsiness Detection System 🚗💤

A computer vision-based system that detects driver fatigue in real time using Eye Aspect Ratio (EAR) and facial landmark detection.

---

## 📌 Overview

This project monitors a driver's eyes through a webcam and detects signs of drowsiness.

If the driver's eyes remain closed for a certain duration, the system triggers an alert to help prevent accidents caused by fatigue.

Developed as part of an entrepreneurship class project.

---

## ✨ Features

- Real-time face and eye tracking
- Eye Aspect Ratio (EAR) based detection
- Drowsiness alert system
- Webcam integration
- Lightweight and simple implementation

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

---

## 🧠 How It Works

1. Detects the driver's face using a webcam
2. Tracks eye landmarks
3. Calculates Eye Aspect Ratio (EAR)
4. Detects prolonged eye closure
5. Triggers an alert when drowsiness is detected

---

## 📂 Project Structure

```txt
driver-drowsiness-detection/
│
├── models/
│   └── face_landmarker.task
│
├── screenshots/
├── drowsiness_detection.py
├── requirements.txt
└── README.md
```
---

## 📚 Learning Outcomes

Through this project, I explored:

- Computer vision fundamentals
- Real-time video processing
- Facial landmark detection
- Human-centered safety systems

---

## 🚧 Future Improvements

- Improve detection accuracy
- Integrate alarm customization
- Deploy on embedded systems

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
python main.py
```

The required MediaPipe model file is already included in the repository under the `models/` folder.


