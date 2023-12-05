import pandas as pd
import numpy as np
from test_gesture import hand_detection_gesture
from sklearn.preprocessing import OneHotEncoder
import joblib

def extract_sequence(video_id):
    video_loc = ('Validation/') + str(video_id)
    sequence = hand_detection_gesture(video_loc)
    #print(sequence.shape)
    return sequence

def generate_data(csv_file, samp_num=7040):
    df = pd.read_csv(csv_file)
    partial = (df.head(samp_num))
    labels = np.array(partial['label_id'].tolist())
    encoder = OneHotEncoder(sparse=False)
    labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))
    full_sequence = np.array([extract_sequence(video) for video in partial['video_id']])
    return full_sequence, labels_one_hot

def save(myarr, myarr2):
    # reshaping the array from 3D matrice to 2D matrice.
    arrReshaped = myarr.reshape(myarr.shape[0], -1)
    # saving reshaped array to file.
    np.savetxt("processed__val_sequence.csv", arrReshaped)
    np.savetxt("processed__val_labels.csv", myarr2)
    # retrieving data from file.

def load():
    loadedArr = np.loadtxt("processed_val_sequence.csv")
    loadedArr2 = np.loadtxt("processed_val_labels.csv")
    # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
    # reshaping to get original matrice with original shape.
    loadedOriginal = loadedArr.reshape(loadedArr.shape[0], loadedArr.shape[1] // 63, 63)
    return loadedOriginal, loadedArr2


#my_list = hand_detection_gesture('Train/112')
#print(my_list)

myseq, mylab = generate_data('Validation.csv')
save(myseq, mylab)
print("shape of sequence:", myseq.shape)
print("shape of y labels:", mylab.shape)
print("----------------------")
new, new2 = load()
print("shape of sequence:", new.shape)
print("shape of y labels:", new2.shape)

