from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import os
import numpy as np
import cv2
import math
import mediapipe as mp
# import helper.data_files_manager as dfm

log_dir = os.path.join('training-logs')
tb_callback = TensorBoard(log_dir=log_dir)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.5)


def process_image(image):
    # Convert the BGR image to RGB and process using MediaPipe Hands
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Extract hand landmarks
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark]).flatten()
    return np.zeros(63)  # Assuming 21 landmarks with x, y, z coordinates


def data_generator(csv_file, batch_size=32, target_size=(176, 100), base_dir=None, fit_label_encoder=False):
    data_df = pd.read_csv(csv_file)
    label_encoder = LabelEncoder()
    if fit_label_encoder:
        # Fit the label encoder and save it
        data_df['label'] = label_encoder.fit_transform(data_df['label'])
        joblib.dump(label_encoder, 'label_encoder.pkl')
    else:
        # Load the fitted label encoder and transform labels
        label_encoder = joblib.load('label_encoder.pkl')
        data_df['label'] = label_encoder.transform(data_df['label'])

    # Base directory for Train folder
    # base_dir = os.path.expanduser('~/Downloads/archive/Train')

    while True:
        for i in range(0, len(data_df), batch_size):
            batch_data = data_df[i:i+batch_size]

            batch_images = []
            batch_labels = []

            for _, row in batch_data.iterrows():
                folder_path = os.path.join(base_dir, str(row['video_id']))
                images = []

                for img_file in sorted(os.listdir(folder_path)):
                    img_path = os.path.join(folder_path, img_file)
                    image = cv2.imread(img_path)
                    if image.shape[:2] != target_size:
                        # Resize if different shape
                        image = cv2.resize(image, target_size)
                    processed_image = process_image(image)
                    images.append(processed_image)

                batch_images.append(images)
                batch_labels.append(row['label'])

            # Convert lists to numpy arrays
            X = np.array(batch_images)
            y = to_categorical(np.array(batch_labels),
                               num_classes=len(label_encoder.classes_))

            yield X, y


class Trainer:
    model = Sequential()

    @staticmethod
    def train(train_csv, val_csv, actions, train_num_samples, val_num_samples, batch_size=32, epochs=10):
        # For training data
        train_gen = data_generator(train_csv, batch_size, base_dir=os.path.expanduser(
            '/media/bella/bellssd/jester/Train'), fit_label_encoder=True)

        # For validation data
        val_gen = data_generator(val_csv, batch_size, base_dir=os.path.expanduser(
            '/media/bella/bellssd/jester/Validation'), fit_label_encoder=False)

        num_classes = len(actions)
        steps_per_epoch = math.ceil(train_num_samples / batch_size)
        validation_steps = math.ceil(val_num_samples / batch_size)
        model = Sequential()  # Defining our feedforward neural network

        # adding layeres to our netowrk - total 6 layers - 1 input ; 4 hidden ; 1 output

        model.add(LSTM(64, return_sequences=True,
                  activation='relu', input_shape=(37, 63)))
        # Long-Short Memory layer - 64 Neurons - Activation ReLU
        # Long-Short Memory layer - 128 Neurons - Activation ReLU
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        # Long-Short Memory layer - 64 Neurons - Activation ReLU
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        # Dense layer - 64 Neurons - Activation ReLU
        model.add(Dense(64, activation='relu'))
        # Dense layer - 32 Neurons - Activation ReLU
        model.add(Dense(32, activation='relu'))
        # Dense layer - 3 Neurons - Activation sofmax

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[
                      'categorical_accuracy'])

        # Callback for saving the model
        checkpoint_callback = ModelCheckpoint(
            'action_model_checkpoint.h5',
            monitor='val_loss',  # Changed to monitor validation loss
            verbose=1,
            save_best_only=True,
            mode='min'
        )

        # Optional: Callback for early stopping
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # Changed to monitor validation loss
            patience=5,
            verbose=1
        )

        model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=[tb_callback, checkpoint_callback,
                       early_stopping_callback]
        )
        model.save('action_model_final.h5')


def get_unique_actions_and_length(csv_file):
    data_df = pd.read_csv(csv_file)
    num_samples = len(data_df)
    unique_actions = data_df['label'].unique()
    return unique_actions.tolist(), num_samples


if __name__ == "__main__":
    train_csv = '/media/bella/bellssd/jester/Train.csv'
    val_csv = '/media/bella/bellssd/jester/Validation.csv'
    actions, train_num_samples = get_unique_actions_and_length(train_csv)
    _, val_num_samples = get_unique_actions_and_length(val_csv)
    checkpoint_path = 'action_model_checkpoint.h5'

    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Loading model from checkpoint")
        Trainer.model = load_model(checkpoint_path)
    else:
        print("Checkpoint not found, starting training from scratch")
    Trainer.train(train_csv, val_csv, actions,
                  train_num_samples, val_num_samples)
