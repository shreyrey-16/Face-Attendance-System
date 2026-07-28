# 🎓 Face Recognition Based Attendance System

A real-time Face Recognition Based Attendance System developed using **Python**, **OpenCV**, and the **LBPH (Local Binary Pattern Histogram)** algorithm. The system automates attendance by recognizing registered users through a webcam and records their **entry** and **exit** timestamps in a CSV file.

This project was developed as part of the **Bachelor of Engineering in Robotics and Artificial Intelligence** at **Thapar Institute of Engineering and Technology**. 

---

# 📌 Project Overview

Traditional attendance methods are time-consuming, error-prone, and vulnerable to proxy attendance. This project replaces manual attendance with a contactless facial recognition system capable of recognizing registered individuals in real time.

The system detects faces using Haar Cascade, recognizes them using the LBPH algorithm, and automatically logs attendance with timestamps.

---

# ✨ Features

- 📷 Real-time face detection using webcam
- 😀 Face recognition using LBPH
- 📝 Automatic attendance logging
- ⏰ Entry and Exit time recording
- 👤 Unknown face detection
- 📊 CSV attendance report generation
- ⚡ Lightweight implementation suitable for CPU

---

# 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- Haar Cascade Classifier
- LBPH Face Recognizer
- Pickle
- CSV

---

# 🧠 Algorithm Used

## Face Detection

The system uses OpenCV's **Haar Cascade Classifier** to detect frontal faces from webcam frames.

## Face Recognition

Detected faces are recognized using the **Local Binary Pattern Histogram (LBPH)** algorithm, which performs well under varying lighting conditions while remaining computationally efficient.

## Model Training

The model is trained using grayscale facial images collected from registered users.

## Attendance Logging

Once a face is recognized:

- IN time is recorded
- OUT time is recorded when the person leaves
- Attendance is saved automatically into a CSV file

---

# 📂 Dataset

A custom dataset is created using the `dataset_collection.py` script.

Dataset Characteristics:

- Images captured using laptop webcam
- 30 images per person
- Grayscale images
- Resolution: 200 × 200 pixels
- Faces extracted using Haar Cascade

> **Privacy Notice**
>
> The dataset and trained models are **not included** in this repository because they contain personal biometric information. Users can create their own dataset using the provided scripts.

---

# 📁 Project Structure

```
Face-Attendance-System/
│
├── dataset_collection.py
├── train_model.py
├── face_recognizer_attendance_inout.py
├── requirements.txt
├── README.md
├── .gitignore
├── trainer/
├── dataset/
└── attendance.csv
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Face-Attendance-System.git
```

Move into the project directory

```bash
cd Face-Attendance-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

## Step 1

Capture face dataset

```bash
python dataset_collection.py
```

## Step 2

Train the recognizer

```bash
python train_model.py
```

## Step 3

Start attendance system

```bash
python face_recognizer_attendance_inout.py
```

---

# 📈 Performance

Under controlled lighting conditions, the system achieved:

| Metric | Result |
|---------|---------|
| Recognition Accuracy | 90–95% |
| Recognition Method | LBPH |
| Face Detection | Haar Cascade |
| Real-time Performance | ~10 FPS |
| Attendance Logging | CSV |

---

# 🔒 Privacy

To protect user privacy, this repository intentionally excludes:

- Face image dataset
- Attendance CSV files
- Trained model files
- Personal information

Users can generate their own dataset using the provided scripts.

---

# 🚀 Future Improvements

- Deep learning-based recognition (FaceNet, Dlib, ResNet)
- GUI/Desktop application
- Cloud database integration
- Attendance analytics dashboard
- Mask detection
- Low-light face recognition

---

# 👩‍💻 Author

**Shreya**

B.E. Robotics & Artificial Intelligence

Thapar Institute of Engineering and Technology

---
