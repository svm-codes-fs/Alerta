"""Evaluates every trained model on data/test (held-out subjects) and writes
results/: metrics table, confusion matrices and accuracy/loss curves.

Positive class for precision/recall/F1 = 'Closed' (the class that raises the alarm).
"""
import json, os, sys
import h5py
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, Input

CLASSES = ['Closed', 'Open']
RUNS = {  # label -> model file
    'Original cnnCat2.h5 (no retraining)': 'models/cnnCat2.h5',
    'Baseline (retrained)': 'models/baseline.keras',
    'Augmented': 'models/augmented.keras',
    'Learning rate 0.0001': 'models/lr0001.keras',
}


def load_original(path):
    # cnnCat2.h5 was saved with Keras 2.1.6 and its config no longer loads in Keras 3,
    # so rebuild the same architecture from model.py and copy the stored weights in.
    m = keras.Sequential([Input((24, 24, 1)), Conv2D(32, 3, activation='relu'), MaxPooling2D(1),
                          Conv2D(32, 3, activation='relu'), MaxPooling2D(1), Conv2D(64, 3, activation='relu'),
                          MaxPooling2D(1), Dropout(.25), Flatten(), Dense(128, activation='relu'), Dropout(.5),
                          Dense(2, activation='softmax')])
    w = h5py.File(path, 'r')['model_weights']
    m.set_weights([w[n][n][k][()] for n in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'dense_1', 'dense_2']
                   for k in ('kernel:0', 'bias:0')])
    return m


test = keras.utils.image_dataset_from_directory('data/test', shuffle=False, color_mode='grayscale',
                                                label_mode='int', image_size=(24, 24), batch_size=256)
X = np.concatenate([x for x, _ in test]) / 255.
y = np.concatenate([y for _, y in test])
print('test images:', len(y), dict(zip(CLASSES, np.bincount(y))))

os.makedirs('results', exist_ok=True)
rows = []
for label, path in RUNS.items():
    if not os.path.exists(path):
        print('skip (not trained yet):', path); continue
    model = load_original(path) if path.endswith('.h5') else keras.models.load_model(path)
    pred = np.argmax(model.predict(X, verbose=0), axis=-1)
    rows.append([label, accuracy_score(y, pred), precision_score(y, pred, pos_label=0),
                 recall_score(y, pred, pos_label=0), f1_score(y, pred, pos_label=0)])
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=CLASSES).plot(cmap='Blues', values_format='d')
    plt.title(label); plt.tight_layout()
    plt.savefig('results/cm_' + os.path.splitext(os.path.basename(path))[0] + '.png', dpi=120); plt.close()

table = '| Model | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|\n' + ''.join(
    f'| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |\n' for r in rows)
print(table)
open('results/comparison.md', 'w').write(table)

# training curves for every run that saved a history
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name in ['baseline', 'augmented', 'lr0001']:
    f = f'results/history_{name}.json'
    if not os.path.exists(f): continue
    h = json.load(open(f))
    for i, k in enumerate(['accuracy', 'loss']):
        line, = ax[i].plot(h[k], '--', label=f'{name} train')
        ax[i].plot(h['val_' + k], color=line.get_color(), label=f'{name} valid')
for i, k in enumerate(['Accuracy', 'Loss']):
    ax[i].set_title(k); ax[i].set_xlabel('epoch'); ax[i].legend(fontsize=8); ax[i].grid(alpha=.3)
plt.tight_layout(); plt.savefig('results/training_curves.png', dpi=120)
