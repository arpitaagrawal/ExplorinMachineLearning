import numpy as np
BLOB_FILENAME = 'hw5_blob.csv'
CIRCLE_FILENAME = 'hw5_circle.csv'
blob_train_data = []
circle_train_data = []

def read_data():
    global blob_train_data, circle_train_data
    blob_train_data = np.genfromtxt(BLOB_FILENAME, delimiter=',')
    circle_train_data = np.genfromtxt(CIRCLE_FILENAME, delimiter=',')
