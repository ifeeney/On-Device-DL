import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from test_gesture import hand_detection_gesture
from sklearn.preprocessing import OneHotEncoder
import pickle


def load():
    loadedArr = np.loadtxt("processed_sequence.csv")
    loadedArr2 = np.loadtxt("processed_labels.csv")
    # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
    # reshaping to get original matrice with original shape.
    loadedOriginal = loadedArr.reshape(loadedArr.shape[0], loadedArr.shape[1] // 63, 63)
    return loadedOriginal, loadedArr2

def load_val():
    loadedArr = np.loadtxt("processed__val_sequence.csv")
    loadedArr2 = np.loadtxt("processed__val_labels.csv")
    # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
    # reshaping to get original matrice with original shape.
    loadedOriginal = loadedArr.reshape(loadedArr.shape[0], loadedArr.shape[1] // 63, 63)
    return loadedOriginal, loadedArr2

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu', input_shape=(37,63)))
    model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(64, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(27, activation='softmax'))
    model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


X_train, y_train = load()
X_test, y_test = load_val()

model = create_model()
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

model.summary()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[cp_callback], batch_size=32, epochs=50)

# Save the entire model to a HDF5 file
model.save('bellas_model3.h5')

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)