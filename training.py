import numpy as np
import tensorflow.keras as k
from data import Film, Frame, analyseAndSaveTimeRange, DataGenerator
from config import rawDataDir, processedDataDir, tfDataDir
import datetime as dt
import time
import plotting as p
import os


modelName = "SimpleFullyConnected"
batchSize = 10
nrBatchesPerEpoch = 100
nrEpochs = 20
validationSteps = 10
nrValidationSamples = 50
timeSteps = int(5 * 60 / 5)
imageSize = 100
imageWidth = imageSize
imageHeight = imageSize
channels = 1


# creating trainingdata
trainingStart = dt.datetime(2016, 6, 1)
trainingEnd = dt.datetime(2016, 6, 30)
validationStart = dt.datetime(2016, 7, 1)
validationEnd = dt.datetime(2016, 7, 15)


analyseAndSaveTimeRange(dt.datetime(2016, 6, 11), validationEnd, 4)

# getting generators
training_generator = DataGenerator(processedDataDir, trainingStart, trainingEnd, nrBatchesPerEpoch, batchSize, timeSteps, False)
validation_generator = DataGenerator(processedDataDir, validationStart, validationEnd, nrBatchesPerEpoch, batchSize, timeSteps, False)


model = k.models.Sequential([
    k.layers.Flatten(input_shape=(timeSteps, imageHeight, imageWidth, channels)),
    k.layers.Dense(50, name="dense1", activation=k.activations.sigmoid),
    k.layers.Dense(50, name="dense2", activation=k.activations.sigmoid),
    k.layers.Dropout(0.2),
    k.layers.Dense(50, name="dense3", activation=k.activations.sigmoid),
    k.layers.Dense(50, name="dense4", activation=k.activations.sigmoid),
    k.layers.Dropout(0.2),
    k.layers.Dense(50, name="dense5", activation=k.activations.sigmoid),
    k.layers.Dense(4, name="dense6", activation=k.activations.softmax)
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
    verbose=2,
    callbacks=[modelSaver, tensorBoard],
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=4,
)


resultDir = f"{tfDataDir}/{modelName}_{int(time.time())}"
os.makedirs(resultDir)

model.save(f"{tfDataDir}/latestRadPredModel.h5")
model.save(f"{resultDir}/{modelName}.h5")
np.savez(f"{resultDir}/history.npz", history.history)

p.saveMultiPlot(f"{resultDir}/loss.png",
                [history.history['loss'], history.history['val_loss']],
                "epoch", "loss", "loss",
                ["training", "validation"])

p.saveMultiPlot(f"{resultDir}/accuracy.png",
                [history.history['categorical_accuracy'], history.history['val_categorical_accuracy']],
                "epoch", "acuracy", "acuracy",
                ["training", "validation"])
