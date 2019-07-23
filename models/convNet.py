import numpy as np
import tensorflow.keras as k
from data import Film, Frame, analyseAndSaveTimeRange, DataGenerator, frameHeight, frameWidth, frameLength
from config import rawDataDir, processedDataDir, tfDataDir, trainingStart, trainingEnd, validationStart, validationEnd





model = k.Sequential([
    k.layers.Conv3D(50, (9, 9, 9), activation="relu", input_shape=(timeSteps, frameHeight, frameWidth, channels)),
    k.layers.MaxPooling3D(pool_size=(5, 5, 5), data_format="channels_last"),
    k.layers.Flatten(),
    k.layers.Dense(30, name="dense1", activation=k.activations.sigmoid),
    k.layers.Dense(4, name="dense2", activation=k.activations.softmax)
])


model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.categorical_crossentropy,
    metrics=[k.metrics.categorical_accuracy]
)

model.summary()