# import cv2

# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer.yml')
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.2, 5)

#     for (x, y, w, h) in faces:
#         id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#         text = f"User {id}" if confidence < 50 else "Unknown"
#         cv2.putText(frame, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load label mapping
with open("labels.pickle", "rb") as f:
    label_ids = pickle.load(f)
    labels = {v: k for k, v in label_ids.items()}  # reverse

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        name = labels.get(id_, "Unknown")

        if confidence < 90:
            label = f"{name} ({int(confidence)}%)"
        else:
            label = "Unknown"

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
