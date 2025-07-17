import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained mask detector model
model = load_model('mask_detector.keras')  # change filename if different

# Load OpenCV's pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define labels and colors
labels = ['Mask', 'No Mask']
colors = [(0, 255, 0), (0, 0, 255)]  # Green for Mask, Red for No Mask

# Start video capture from default webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract face ROI and preprocess for model
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (224, 224))
        face_array = face_resized.astype("float32") / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # Predict mask/no mask
        prediction = model.predict(face_array)
        class_idx = np.argmax(prediction)
        label = labels[class_idx]
        color = colors[class_idx]

        # Display label and bounding box
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show the frame
    cv2.imshow("Real-Time Face Mask Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
