# Tests the hand landmarker feature extraction on a video from the test dataset

import tflite_runtime.interpreter as tflite
import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

frame_rate = 14
prev_time = 0
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

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
            
        if type == 'LIVE_STREAM':
            # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
            options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path="Downloads/hand_landmarker.task"), # path to model
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
            num_hands = 1, # track both hands
            min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
            min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
            min_tracking_confidence = 0.3, # lower than value to get predictions more often
            result_callback=update_result)
        elif type == 'IMAGE':
            options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path="Downloads/hand_landmarker.task"), # path to model
            running_mode = mp.tasks.vision.RunningMode.IMAGE, # running on a live stream
            num_hands = 1, # track both hands
            min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
            min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
            min_tracking_confidence = 0.3) # lower than value to get predictions more often

        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)
        
    def detect(self, frame):
        # Load the input image from an image file.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_landmarker_result = self.landmarker.detect(image)
        return image, hand_landmarker_result
    
    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        # close landmarker
        self.landmarker.close()

def hand_detection_livestream(predict=False, port: int = 0):

    # Create a hand landmarker instance with the live stream mode:
    vid_landmarker = landmarker_and_result('LIVE_STREAM')
    prev_time = 0
    returned = []
    
    if(predict):
        img_landmarker = landmarker_and_result('IMAGE')
        sequence = []
        interpreter = tflite.Interpreter('Downloads/bellas_model5.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    
    #Attempt to open camera
    cap = cv2.VideoCapture(port)
    if not cap.isOpened():  # Check if the web cam is opened correctly
        print("failed to open cam")
    else:
        print('cam opened on port {}'.format(port))

    while(True):

        # Capture the video frame 
        success, frame = cap.read()
        if not success:
            print('failed to capture frame')
            break

        vid_landmarker.detect_async(frame)
        
        # Capture frames for prediction at a specified framerate
        time_elapsed = time.time() - prev_time
        if(predict and (time_elapsed > 1./frame_rate)):
            prev_time = time.time()            
            resized = cv2.resize(frame, (176, 100), interpolation=cv2.INTER_AREA)
            image, detection_result = img_landmarker.detect(resized)
            # Record coordinates for each landmark
            curr_keypoints = []
            hand_landmarks_list = detection_result.hand_world_landmarks
            if len(hand_landmarks_list) == 0:
                for j in range(0, 21):
                    curr_keypoints.extend([0.0, 0.0, 0.0])
            else:
                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]
                    for landmarks in hand_landmarks:
                        curr_keypoints.extend([landmarks.x,landmarks.y, landmarks.z])
            # Add current keypoints to running sequence
            sequence.append(curr_keypoints)
            # Keep only the most recent 37 frames
            sequence = sequence[-37:]
            
            # If we have enough frames, perform a prediction
            if len(sequence) == 37:
                #prepare data for prediction
                reshaped = np.array(sequence) #[np.zeros_like(sequence[0])] * 10 + sequence + [np.zeros_like(sequence[0])] * 7)
                reshaped = reshaped.reshape((1, 37, 63))
                
                # Run tflite model
                interpreter.set_tensor(input_details[0]["index"], np.float32(reshaped))
                interpreter.invoke()
                result = interpreter.get_tensor(output_details[0]["index"])
                interpreter.reset_all_variables()
                
                # Return predictions
                returned = []
                top3_indices = np.argsort(result[0])[-3:]
                for index in top3_indices:
                    returned.append(f"{class_labels[index]}, Prob: {result[0][index]}")
                print(class_labels[np.argmax(result, axis=-1).item()])
        
        frame = draw_landmarks_on_image(frame,returned,vid_landmarker.result)
        cv2.imshow('',frame)

        # the 'q' button is set as the quit button
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    
def draw_landmarks_on_image(rgb_image, probs, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Taken from https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
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
            
            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, probs[2],(30, 30), cv2.FONT_HERSHEY_DUPLEX,0.7, (59, 240, 23), FONT_THICKNESS, cv2.LINE_AA)
            cv2.putText(annotated_image, probs[1],(30, 60), cv2.FONT_HERSHEY_DUPLEX,0.7, (240, 226, 23), FONT_THICKNESS, cv2.LINE_AA)
            cv2.putText(annotated_image, probs[0],(30, 90), cv2.FONT_HERSHEY_DUPLEX,0.7, (240, 81, 23), FONT_THICKNESS, cv2.LINE_AA)

         return annotated_image
   except:
      return rgb_image

hand_detection_livestream(predict = True)