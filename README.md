# On-Device-Deep Learning

## Files used: 

### Initial Training
- `trainer.py` was used to train our model initially
- `inference.py` was used to test the model
- 
### Revised Training 
- `test_gesture.py` is used to gather landmark coordinates for sequences of images using Mediapipe Hand Landmarker
  - see https://developers.google.com/mediapipe/solutions/vision/hand_landmarker for more details
- `prepare_data.py` is used to convert the jester dataset image sequences into landmarks returned in test_gesture.py
- `model.py` is used to train a LSTM model on the prepared data
  - model is saved to a checkpoint every 5 epochs
  - model history is saved to trainHistoryDict
- `Evaluate_models.ipynb` is used to load model history and visualize/benchmark the performance
  - bella_model.h5 is saved after the training is complete
  - pruned_and_quantized.tflite is then created with a signature specified to enable use in the tflite runtime
    
### On-Device
- `test_tflite.py` is used to test the accuracy of the tensorflow lite model on Jester test data
- `single_gesture_classifier.py` is used to collect a 3 single second gesture and then perform classification
- `real_time_classifier.py` is used to perform continuous gesture classification in real-time
## Performance(Inference.py) after initial training using trainer.py
![Stop SIgn](/Media/Shivam_stop_sign.png)
![Thumbs up](Media/Shivam_thumbs_up.png)
![Video](Media/Shivam_Hand_gesture_recognition-ezgif.com-video-to-gif-converter.gif)



## Demo Videos
Performing asynchronous gesture classification, using the file 'Test_tflite.py':

https://github.com/ifeeney/On-Device-DL/assets/42654829/86a2c5c6-f2be-4969-9f64-e54f49f6d016

Performing real-time gesture classification, using the file 'Test_gesture.py':

https://github.com/ifeeney/On-Device-DL/assets/42654829/d22b8909-e587-4798-b10d-c730735ed4ac

https://github.com/ifeeney/On-Device-DL/assets/42654829/8b85f7e6-3592-4890-ae0d-b209075000f4

## Authorship
Shivam Sharma: `trainer.py`, `inference.py`

Isabella Feeney: `test_gesture.py`, `prepare_data.py`, `model.py`, `Evaluate_models.ipynb`, `test_tflite.py`, `single_gesture_classifier.py`, `real_time_classifier.py`
