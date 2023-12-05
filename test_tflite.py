import tflite_runtime.interpreter as tflite

import numpy as np

def load_val():
    loadedArr = np.loadtxt("Downloads/processed__val_sequence.csv")
    loadedArr2 = np.loadtxt("Downloads/processed__val_labels.csv")
    # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
    # reshaping to get original matrice with original shape.
    loadedOriginal = loadedArr.reshape(loadedArr.shape[0], loadedArr.shape[1] // 63, 63)
    return loadedOriginal, loadedArr2

X_test, y_test = load_val()

# Run the model with TensorFlow to get expected results.
TEST_CASES = 100
true_count = 0

# Run the model with TensorFlow Lite
interpreter = tflite.Interpreter('Downloads/bellas_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i in range(TEST_CASES):
  #print("Begin")
  #print(X_test[i:i+1])
  #print(input_details)
  interpreter.set_tensor(input_details[0]["index"], np.float32(X_test[i:i+1]))
  interpreter.invoke()
  result = interpreter.get_tensor(output_details[0]["index"])

  # Assert if the result of TFLite model is consistent with the TF model.
  target = np.argmax(y_test[i], axis=-1)
  returned = np.argmax(result, axis=-1)
  
  #print(target)
  #print(returned)
  if(target == returned):
      true_count += 1
      print("True")
  else:
      print("False")

  # Please note: TfLite fused Lstm kernel is stateful, so we need to reset
  # the states.
  # Clean up internal states.
  interpreter.reset_all_variables()
  
print("Matched:", true_count)