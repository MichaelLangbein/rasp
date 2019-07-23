import numpy as np
import tensorflow.keras as k
from data import Film, Frame, analyseAndSaveTimeRange, DataGenerator, frameHeight, frameWidth, frameLength
from config import rawDataDir, processedDataDir, tfDataDir, trainingStart, trainingEnd, validationStart, validationEnd





input_tensor = k.Input(shape=(timeSteps, frameHeight, frameWidth, channels))

pool1 = k.layers.MaxPooling3D(pool_size=(1, 100, 100), data_format="channels_last")(input_tensor)
resh1 = k.layers.Reshape((60, 1))(pool1)
lstm1 = k.layers.LSTM((30))(resh1)

conv2 = k.layers.Conv3D(100, (9, 9, 9), activation="relu")(input_tensor)
pool2 = k.layers.MaxPooling3D(pool_size=(1, 92, 92), data_format="channels_last")(conv2)
resh2 = k.layers.Reshape((52, 100))(pool2)
lstm2 = k.layers.LSTM((30))(resh2)

conv3a = k.layers.Conv3D(100, (9, 9, 9), activation="relu")(input_tensor)
conv3b = k.layers.Conv3D(100, (9, 9, 9), activation="relu")(conv3a)
pool3 = k.layers.MaxPooling3D(pool_size=(1, 84, 84), data_format="channels_last")(conv3b)
resh3 = k.layers.Reshape((44, 100))(pool3)
lstm3 = k.layers.LSTM((30))(resh3)

merged = k.layers.concatenate([lstm1, lstm2, lstm3], axis=-1)
denseA = k.layers.Dense(30, name="dense1", activation=k.activations.sigmoid)(merged)
denseB = k.layers.Dense(4, name="dense2", activation=k.activations.softmax)(denseA)

model = k.models.Model(input_tensor, denseB)


model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.categorical_crossentropy,
    metrics=[k.metrics.categorical_accuracy]
)

model.summary()