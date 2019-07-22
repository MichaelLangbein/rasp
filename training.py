import os
import sys
import time
import datetime as dt
import numpy as np
import tensorflow.keras as k
from data import Film, Frame, analyseAndSaveTimeRange, DataGenerator, frameHeight, frameWidth, frameLength
from config import rawDataDir, processedDataDir, tfDataDir
import plotting as p


modelName = "SimpleConv3DGruNet"
batchSize = 20
nrBatchesPerEpoch = 30
nrEpochs = 5
timeSteps = int(5 * 60 / 5)
timeSeriesOffset = 5
channels = 1


# creating trainingdata
trainingStart = dt.datetime(2016, 6, 1)
trainingEnd = dt.datetime(2016, 6, 30)
validationStart = dt.datetime(2016, 7, 1)
validationEnd = dt.datetime(2016, 7, 15)


# getting generators
training_generator = DataGenerator(processedDataDir, trainingStart, trainingEnd, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)
validation_generator = DataGenerator(processedDataDir, validationStart, validationEnd, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)


model = k.models.Sequential([
    k.layers.Conv3D(100, (9, 9, 9), name="conv1", activation='relu', input_shape=(timeSteps, frameHeight, frameWidth, channels)),  # learn 100 tempo-spatial features
    k.layers.MaxPooling3D(pool_size=(1, 92, 92), data_format="channels_last"),  # only return wheather a feature is or is not present
    k.layers.Reshape((52, 100)),  # 52 timesteps of 100 features each
    k.layers.LSTM(30),  # return 30 labels once full timeseries has been seen
    #  k.layers.Flatten(),
    k.layers.Dense(30, name="dense1", activation=k.activations.sigmoid),
    k.layers.Dense(4, name="dense2", activation=k.activations.softmax)
])


model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.categorical_crossentropy,
    metrics=[k.metrics.categorical_accuracy]
)

print(model.summary())


modelSaver = k.callbacks.ModelCheckpoint(
    tfDataDir + f"{modelName}_checkpoint.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True
)

tensorBoard = k.callbacks.TensorBoard(
    log_dir=f"{tfDataDir}/tensorBoardLogs",
    histogram_freq=3,
    batch_size=32,
    write_graph=True,
    write_grads=False,
    write_images=False
)


history = model.fit_generator(
    training_generator,
    epochs=nrEpochs,
    verbose=1,
    callbacks=[modelSaver, tensorBoard],
    validation_data=validation_generator,
    use_multiprocessing=False
    # use_multiprocessing=True,
    # workers=4,
)


resultDir = f"{tfDataDir}/{modelName}_{int(time.time())}"
os.makedirs(resultDir)

model.save(f"{tfDataDir}/latestRadPredModel.h5")
model.save(f"{resultDir}/{modelName}.h5")
np.savez(f"{resultDir}/history.npz", history.history)

fig, ax = p.plotMultiPlot([history.history['loss'], history.history['val_loss']],
                          "epoch", "loss", "loss",
                          ["training", "validation"])
fig.save(f"{resultDir}/loss.png")

fig, ax = p.plotMultiPlot([history.history['categorical_accuracy'], history.history['val_categorical_accuracy']],
                          "epoch", "acuracy", "acuracy",
                          ["training", "validation"])
fig.save(f"{resultDir}/accuracy.png")

fig, ax = p.plotValidationHitGrid(model, validation_generator, 500)
fig.save(f"{resultDir}/hitGrid.png")

with open(f"{resultDir}/{modelName}_description.txt", "w") as f:
    f.write(model.summary())
