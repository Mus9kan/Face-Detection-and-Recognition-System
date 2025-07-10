# import cv2
# import numpy as np
# from PIL import Image
# import os

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# path = 'dataset'

# def get_images_and_labels(path):
#     face_samples = []
#     ids = []

#     for image_file in os.listdir(path):
#         img_path = os.path.join(path, image_file)
#         gray_img = Image.open(img_path).convert('L')
#         img_np = np.array(gray_img, 'uint8')
#         id = int(image_file.split('.')[1])

#         faces = face_detector.detectMultiScale(img_np)
#         for (x, y, w, h) in faces:
#             face_samples.append(img_np[y:y+h, x:x+w])
#             ids.append(id)

#     return face_samples, ids

# print("[INFO] Training faces...")
# faces, ids = get_images_and_labels(path)
# recognizer.train(faces, np.array(ids))
# recognizer.write('trainer.yml')
# print("[INFO] Model training complete. Saved as 'trainer.yml'")
import cv2
import numpy as np
from PIL import Image
import os
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = "dataset"
label_ids = {}
current_id = 0
face_samples = []
ids = []

# Build mapping and prepare data
for image_file in os.listdir(dataset_path):
    if not image_file.endswith(".jpg"):
        continue

    name = image_file.split('.')[0]

    if name not in label_ids:
        label_ids[name] = current_id
        current_id += 1

    id_ = label_ids[name]
    img_path = os.path.join(dataset_path, image_file)
    gray_img = Image.open(img_path).convert('L')
    img_np = np.array(gray_img, 'uint8')

    faces = face_detector.detectMultiScale(img_np)
    for (x, y, w, h) in faces:
        face_samples.append(img_np[y:y+h, x:x+w])
        ids.append(id_)

# Save labels
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(face_samples, np.array(ids))
recognizer.write("trainer.yml")
print("[INFO] Model trained and label mapping saved.")
