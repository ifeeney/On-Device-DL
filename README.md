# On-Device-DL

## Currently used files
- `test_gesture.py` is used to gather landmark coordinates for sequences of images using Mediapipe Hand Landmarker
  - see https://developers.google.com/mediapipe/solutions/vision/hand_landmarker for more details
- `prepare_data.py` is used to convert the jester dataset image sequences into landmarks returned in test_gesture.py
- `model.py` is used to train a LSTM model on the prepared data
  - model is saved to a checkpoint every 5 epochs
  - model history is saved to trainHistoryDict
- `Train_gestures.ipynb` is used to load model history and visualize/benchmark the performance
  - bella_model.h5 is saved after the training is complete
  - bella_model.tflite is then created with a signature specified to enable use in the tflite runtime
- `Test_tflite.py` can then be run on a raspberry pi to perform gesture classification
