import cv2
import os


def collect_dataset(person_name, count=30):

    save_path = f'dataset/{person_name}'
    os.makedirs(save_path, exist_ok=True)
    

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    

    image_id = 0
    print(f"[INFO] Collecting samples for {person_name}...")

    while True:
        ret, frame = cam.read()
       
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            image_id += 1
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            cv2.imwrite(f"{save_path}/{image_id}.jpg", face_img)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif image_id >= count:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Dataset collection completed: {image_id} images.")

# Usage
if __name__ == "__main__":
    name = input("Enter person's name: ").strip().lower()
    collect_dataset(name, count=30)

