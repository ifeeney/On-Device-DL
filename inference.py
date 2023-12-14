import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
​
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
​
​
def process_image(image):
    # Convert the BGR image to RGB and process using MediaPipe Hands
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
​
    # Extract hand landmarks
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark]).flatten()
    return np.zeros(63)  # Assuming 21 landmarks with x, y, z coordinates
​
​
def recreate_label_encoder(csv_file):
    data_df = pd.read_csv(csv_file)
    label_encoder = LabelEncoder()
    label_encoder.fit(data_df['label'])
    return label_encoder
​
​
def process_frame(frame, target_size=(176, 100)):
    # Resize frame
    frame_resized = cv2.resize(frame, target_size)
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Process frame with MediaPipe
    results = hands.process(frame_rgb)
​
    # Check if any landmarks are detected and draw them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)
​
    # Process frame (similar to process_image in your training code)
    processed_frame = process_image(frame_rgb)
    return processed_frame, results
​
​
# Load model
model_path = '/home/shivam/Real-Time-Hand-Gesture-Recognition/action_model_checkpoint.h5'
# or 'action2.h5'
model = load_model(model_path)
​
# Recreate label encoder
# or any other CSV file you need
csv_file = '/home/shivam/Downloads/archive/Train.csv'
label_encoder = recreate_label_encoder(csv_file)
​
​
# Start video capture
cap = cv2.VideoCapture(0)
​
# Buffer to store a sequence of frames
frame_buffer = np.zeros((37, 63))
​
while True:
    ret, frame = cap.read()
    if not ret:
        break
​
    # Process the frame and get landmarks
    processed_frame, results = process_frame(frame)
​
    # Update the frame buffer
    frame_buffer = np.roll(frame_buffer, -1, axis=0)
    frame_buffer[-1] = processed_frame
​
    # Check if buffer is full (i.e., contains 37 frames)
    if np.count_nonzero(frame_buffer) == frame_buffer.size:
        # Make prediction using the buffer
        prediction = model.predict(np.expand_dims(frame_buffer, axis=0))
        predicted_class = label_encoder.inverse_transform(
            [np.argmax(prediction)])[0]
​
        # Display the prediction on the frame
        cv2.putText(frame, predicted_class, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
​
    # Show the frame with MediaPipe hand landmarks
    cv2.imshow('Frame', frame)
​
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
​
# Release the video capture object
cap.release()
cv2.destroyAllWindows()
