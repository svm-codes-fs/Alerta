import os, json, argparse
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, Input

# --- experiment switches (defaults = original project setup) ---
ap = argparse.ArgumentParser()
ap.add_argument('--name', default='baseline')                 # saved as models/<name>.keras
ap.add_argument('--augment', action='store_true')              # Experiment 2: data augmentation
ap.add_argument('--lr', type=float, default=0.001)             # Experiment 3: learning rate (Adam default = 0.001)
ap.add_argument('--epochs', type=int, default=15)
args = ap.parse_args()
keras.utils.set_random_seed(42)

# Keras 3 removed ImageDataGenerator/flow_from_directory; this is the equivalent loader
# (grayscale, 24x24, rescale 1/255, one-hot labels, classes = folder names: Closed=0, Open=1)
def generator(dir, shuffle=True,batch_size=1,target_size=(24,24)):
    ds = keras.utils.image_dataset_from_directory(dir, batch_size=batch_size, shuffle=shuffle, seed=42,
            color_mode='grayscale', label_mode='categorical', image_size=target_size)
    return ds.map(lambda x, y: (x / 255., y))

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)

# Experiment 2: random rotation / zoom / brightness / horizontal flip, applied to TRAINING images only
if args.augment:
    augment = Sequential([
        keras.layers.RandomRotation(10/360),               # +-10 degrees
        keras.layers.RandomZoom(0.1),                      # +-10 %
        keras.layers.RandomBrightness(0.2, value_range=(0, 1)),
        keras.layers.RandomFlip('horizontal'),             # a mirrored eye is still a valid eye
    ])
    train_batch = train_batch.map(lambda x, y: (augment(x, training=True), y))
train_batch = train_batch.prefetch(tf.data.AUTOTUNE)


model = Sequential([
    Input((24,24,1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling

#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :)
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),loss='categorical_crossentropy',metrics=['accuracy'])

hist = model.fit(train_batch, validation_data=valid_batch, epochs=args.epochs)

# new runs are saved under their own name so the original models/cnnCat2.h5 is kept untouched
model.save(f'models/{args.name}.keras')
os.makedirs('results', exist_ok=True)
json.dump(hist.history, open(f'results/history_{args.name}.json', 'w'))
