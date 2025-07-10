# import cv2
# import os

# # Create dataset folder if not exists
# if not os.path.exists("dataset"):
#     os.makedirs("dataset")

# # Input user ID
# user_id = input("Enter user ID: ")

# # Initialize webcam and face detector
# cap = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         count += 1
#         face = gray[y:y+h, x:x+w]
#         cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", face)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     cv2.imshow("Capturing Faces", frame)

#     if cv2.waitKey(1) == ord('q') or count >= 50:
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("[INFO] Dataset collection complete.")
import cv2
import os

name = input("Enter your name: ").strip()

dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        filename = f"{dataset_path}/{name}.{count}.jpg"
        cv2.imwrite(filename, face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Collected {count} face images for '{name}'")
