import cv2 
import os 
import numpy as np
import pickle

def train_model(dataset_dir='dataset', model_path='trainer.yml', label_path='labels.pkl'):
    '''train_model(...): Defines a function that will train the face recognition model.

dataset_dir='dataset': The folder where your training images are stored.

model_path='trainer.yml': The file where the trained model will be saved.

label_path='labels.pkl': The file where the mapping between person names and IDs will be saved.

'''
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []

    print("[INFO] Starting training...")

    for root, dirs, files in os.walk(dataset_dir):
        #Walks through the folder (dataset/) and gets each subdirectory (person_name/) and the image files inside.
        for file in files:#Loops through each file in the dataset. Only processes images with .jpg or .png extensions.
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)#path: Full path to the image file.
                label = os.path.basename(root).lower().replace(" ", "_")#label: The name of the person (folder name), formatted to lowercase with underscores.

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                    #If a label (person) is seen for the first time, assign it a new numeric ID.Store the ID of the current image.
                id_ = label_ids[label]

                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Reads the image in grayscale (helps the model learn better and is faster).Skips the image if it can't be read.
                if image is None:
                    continue

                # Detect and crop face again just in case
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
                #Detects faces in the image using the cascade classifier.
                for (x, y, w, h) in faces:
                    roi = image[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi, (200, 200))
                    #(x, y, w, h): Coordinates of the detected face.
                    #roi: Region of Interest, i.e., the face area.roi_resized: Resize face to a fixed size (200x200 pixels) for consistent training.
                    x_train.append(roi_resized)
                    y_labels.append(id_)
                    #Add the face and its label (ID) to the training data.



    print(f"[INFO] Number of training samples: {len(x_train)}")
    recognizer.train(x_train, np.array(y_labels))
    #Tells how many face images were collected.

    #Trains the model using the faces and their labels.



    recognizer.save(model_path)
    #Saves the trained model to a file (trainer.yml).
    with open(label_path, 'wb') as f:
        pickle.dump(label_ids, f)
        #Saves the label dictionary (label_ids) as a file using pickle, so it can be used later during face recognition.

    print(f"[SUCCESS] Training complete. Model saved to {model_path}, labels to {label_path}")
    #Confirmation message after successful training.

# Run training
if __name__ == "__main__":
    train_model()

