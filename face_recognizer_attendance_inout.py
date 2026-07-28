import cv2
import pickle
import os
from datetime import datetime, timedelta

# Load trained model and label mappings
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}  # Flip id:name

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Attendance tracking
in_times = {}
last_seen = {}
logged_out = set()

# Write to attendance CSV
def log_event(name, in_time, out_time=None):
    with open('attendance.csv', 'a') as f:
        if out_time:
            f.write(f"{name},{in_time},{out_time}\n")
        else:
            f.write(f"{name},{in_time},\n")

# Start video capture
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.now()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    detected_names = set()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (200, 200))

        id_, conf = recognizer.predict(roi_resized)
        if conf < 70:
            name = labels.get(id_)

            detected_names.add(name)

            if name not in in_times:
                in_times[name] = now.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[IN] {name} at {in_times[name]}")
                log_event(name, in_times[name])  # log with empty out_time

            last_seen[name] = now

            color = (0, 255, 0)
            label_text = f"{name} ({int(conf)})"
        else:
            label_text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Check for timeouts (absent for >15 sec)
    for name in list(last_seen.keys()):
        if name not in detected_names and name not in logged_out:
            last_time = last_seen[name]
            if now - last_time > timedelta(seconds=15):
                out_time = now.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[OUT] {name} at {out_time}")
                log_event(name, in_times.get(name, ""), out_time)
                logged_out.add(name)

    cv2.imshow("Face Recognition - In/Out Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Session ended.")
