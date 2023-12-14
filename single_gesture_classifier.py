import cv2
import os
import time
import tflite_runtime.interpreter as tflite
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

port = 0
prev = 0
frame_count = 0
frame_rate = 12
dirname = 'gesture_22'
class_labels = ["Doing other things","Drumming Fingers","No Gesture","Pulling Hand In","Pulling Two Fingers In","Pushing Hand Away","Pushing Two Fingers Away","Rolling Hand Backward","Rolling Hand Forward","Shaking Hand","Sliding Two Fingers Down","Sliding Two Fingers Left","Sliding Two Fingers Up","Sliding Two Fingers Right","Stop Sign","Swiping Down","Swiping Left","Swiping Right","Swiping Up","Thumb Down","Thumb Up","Turning Hand Clockwise","Turning Hand Counterclockwise","Zoomig In With Full Hand","Zooming In With Two Fingers","Zooming Out With Full Hand","Zooming Out With Two Fingers"]

class landmarker_and_result():
    def __init__(self, type):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker(type)
   
    def createLandmarker(self, type):
        # callback function
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions( 
        base_options = mp.tasks.BaseOptions(model_asset_path="Downloads/hand_landmarker.task"), # path to model
        running_mode = mp.tasks.vision.RunningMode.IMAGE, # running on a live stream
        num_hands = 1, # track both hands
        min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
        min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
        min_tracking_confidence = 0.3) # lower than value to get predictions more often
      
        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)
   
    def detect(self, input):
        # Load the input image from an image file.
        image = mp.Image.create_from_file(input)
        hand_landmarker_result = self.landmarker.detect(image)
        return image, hand_landmarker_result

    def close(self):
        # close landmarker
        self.landmarker.close()
    
def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Taken from https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         handedness_list = detection_result.handedness
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())

         return annotated_image
   except:
      return rgb_image

def hand_detection_gesture(folder_path, print_image = False):
    img_list = []
    keypoint_list = [0] * 37
    image_landmarker = landmarker_and_result('IMAGE')

    for i in range (1, 38):
        curr_keypoints = []
        image_path = f"{folder_path}/frame{i}.jpg"
        image, detection_result = image_landmarker.detect(image_path)

        hand_landmarks_list = detection_result.hand_world_landmarks
        #print(hand_landmarks_list)
        if len(hand_landmarks_list) == 0:
            for j in range(0, 21):
                curr_keypoints.extend([0.0, 0.0, 0.0])
        else:
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                for landmarks in hand_landmarks:
                    curr_keypoints.extend([landmarks.x,landmarks.y, landmarks.z])

        keypoint_list[i-1] = curr_keypoints

        if(print_image):
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            img_list.append(annotated_image)
        #cv2.imshow('',annotated_image)
        #cv2.waitKey(0)

    #print(keypoint_list)

    if(print_image):
        # Calculate the number of rows and columns in the grid
        rows = 5  # you can adjust this based on the layout you want
        cols = 8  # you can adjust this based on the layout you want

        # Create a blank canvas (white background)
        canvas = np.ones((rows * img_list[0].shape[0], cols * img_list[0].shape[1], 3), dtype=np.uint8) * 255

        # Populate the canvas with images
        for i in range(min(len(img_list), rows * cols)):
            row_index = i // cols
            col_index = i % cols
            y_offset = row_index * img_list[0].shape[0]
            x_offset = col_index * img_list[0].shape[1]
            canvas[y_offset:y_offset + img_list[0].shape[0], x_offset:x_offset + img_list[0].shape[1]] = img_list[i]

        # Display the composite image
        cv2.imshow('Composite Image', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return(np.array(keypoint_list))

#---------------------------------------------

# Initialize TensorFlow Lite interpreter
interpreter = tflite.Interpreter('Downloads/quantized_and_pruned.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Record a new gesture
cap = cv2.VideoCapture(port)
if not cap.isOpened():  # Check if the web cam is opened correctly
    print("failed to open cam")
else:
    print('cam opened on port {}'.format(port))
    os.mkdir(dirname)
    os.chdir(dirname)

while(frame_count < 37):
    time_elapsed = time.time() - prev
    # Capture the video frame 
    success, frame = cap.read()
    if not success:
        print('failed to capture frame')
        break
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        frame_count += 1
        name = "frame%d.jpg"%frame_count
        resized = cv2.resize(frame, (176, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite(name, resized)
    cv2.imshow('',frame)
    # the 'q' button is set as the quit button
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

# Extract hand landmarks using mediapipe
os.chdir('/home/bella')
print("extracting keypoints")
keypoints = hand_detection_gesture(dirname, False)
reshaped = keypoints.reshape((1, 37, 63))

# Classify gesture using tflite model
print("classifying gesture")
interpreter.set_tensor(input_details[0]["index"], np.float32(reshaped))
interpreter.invoke()
result = interpreter.get_tensor(output_details[0]["index"])

# Return classification
returned = np.argmax(result, axis=-1)
label_idx = int(returned)
print(class_labels[label_idx])

# Clean up internal states.
interpreter.reset_all_variables()
  
