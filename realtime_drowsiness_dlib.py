import cv2
import numpy as np
import dlib
import winsound
from tensorflow.keras.models import load_model
from imutils import face_utils

# Load your trained CNN model
model = load_model('eye_state_cnn_model.h5')

# Load dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks (2).dat')  # Use your exact filename

# Eye landmark indices for left and right eyes from the 68-points model
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

# Parameters
EYE_SIZE = (24, 24)
drowsy_limit = 3  # Lowered to 3 frames for quicker alert during testing

# Variables to count closed eye frames
closed_eyes_frames = 0

def extract_eye(image, landmarks, eye_points):
    points = [landmarks.part(pt) for pt in eye_points]
    x_coords = [pt.x for pt in points]
    y_coords = [pt.y for pt in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    padding = 2
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)

    eye_img = image[y_min:y_max, x_min:x_max]
    eye_img_resized = cv2.resize(eye_img, EYE_SIZE)
    eye_img_normalized = eye_img_resized / 255.0
    eye_img_reshaped = eye_img_normalized.reshape(1, EYE_SIZE[0], EYE_SIZE[1], 1)

    return eye_img_reshaped, (x_min, y_min, x_max, y_max)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    any_eye_closed = False

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_img, left_coords = extract_eye(gray, landmarks, LEFT_EYE_POINTS)
        left_pred = model.predict(left_eye_img, verbose=0)
        left_class = np.argmax(left_pred)
        left_label = "Closed" if left_class == 0 else "Open"
        if left_label == "Closed":
            any_eye_closed = True

        right_eye_img, right_coords = extract_eye(gray, landmarks, RIGHT_EYE_POINTS)
        right_pred = model.predict(right_eye_img, verbose=0)
        right_class = np.argmax(right_pred)
        right_label = "Closed" if right_class == 0 else "Open"
        if right_label == "Closed":
            any_eye_closed = True

        lx_min, ly_min, lx_max, ly_max = left_coords
        rx_min, ry_min, rx_max, ry_max = right_coords

        left_color = (0, 0, 255) if left_label == "Closed" else (0, 255, 0)
        right_color = (0, 0, 255) if right_label == "Closed" else (0, 255, 0)

        cv2.rectangle(frame, (lx_min, ly_min), (lx_max, ly_max), left_color, 2)
        cv2.putText(frame, left_label, (lx_min, ly_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)

        cv2.rectangle(frame, (rx_min, ry_min), (rx_max, ry_max), right_color, 2)
        cv2.putText(frame, right_label, (rx_min, ry_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)

    if any_eye_closed:
        closed_eyes_frames += 1
    else:
        closed_eyes_frames = 0

    if closed_eyes_frames >= drowsy_limit:
        print("DROWSY ALERT Triggered!")  # Debug message in console
        cv2.putText(frame, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        winsound.Beep(1000, 1000)

    cv2.imshow("Drowsiness Detection (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
