"""Loads the original eye-state CNN (models/cnnCat2.h5) in current Keras."""
import os

import h5py
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

from . import ROOT


def load_eye_model(path=os.path.join(ROOT, 'models', 'cnnCat2.h5')):
    # cnnCat2.h5 was saved by Keras 2.1.6 and its config does not load in Keras 3:
    # rebuild the same network as model.py and copy the stored weights in.
    model = Sequential([Input((24, 24, 1)), Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((1, 1)),
                        Conv2D(32, (3, 3), activation='relu'), MaxPooling2D((1, 1)),
                        Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((1, 1)), Dropout(0.25), Flatten(),
                        Dense(128, activation='relu'), Dropout(0.5), Dense(2, activation='softmax')])
    with h5py.File(path, 'r') as f:
        w = f['model_weights']
        model.set_weights([w[n][n][k][()] for n in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'dense_1', 'dense_2']
                           for k in ('kernel:0', 'bias:0')])
    return model
