import os
import sys
import time
import datetime as dt
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from data import Film, Frame, analyseAndSaveTimeRange, DataGenerator, frameHeight, frameWidth, frameLength
from config import rawDataDir, processedDataDir, tfDataDir, trainingStart, trainingEnd, validationStart, validationEnd
import plotting as p


modelName = "ConvLstmInception2"  # CoLIc ?

batchSize = 20
nrBatchesPerEpoch = 30
nrEpochs = 8
timeSteps = int(5 * 60 / 5)
timeSeriesOffset = 5
channels = 1


training_generator = DataGenerator(processedDataDir, trainingStart, trainingEnd, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)
validation_generator = DataGenerator(processedDataDir, validationStart, validationEnd, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)


input_tensor = k.Input(shape=(timeSteps, frameHeight, frameWidth, channels))

pool1 = k.layers.MaxPooling3D(pool_size=(1, 100, 100), data_format="channels_last")(input_tensor)
resh1 = k.layers.Reshape((60, 1))(pool1)
lstm1 = k.layers.LSTM((30))(resh1)

conv2 = k.layers.Conv3D(100, (9, 9, 9), activation="relu")(input_tensor)
pool2 = k.layers.MaxPooling3D(pool_size=(1, 92, 92), data_format="channels_last")(conv2)
resh2 = k.layers.Reshape((52, 100))(pool2)
lstm2 = k.layers.LSTM((30))(resh2)

#  conv3a = k.layers.Conv3D(100, (9, 9, 9), activation="relu")(input_tensor)
#  conv3b = k.layers.Conv3D(100, (9, 9, 9), activation="relu")(conv3a)
#  pool3 = k.layers.MaxPooling3D(pool_size=(1, 84, 84), data_format="channels_last")(conv3b)
#  resh3 = k.layers.Reshape((44, 100))(pool3)
#  lstm3 = k.layers.LSTM((30))(resh3)

merged = k.layers.concatenate([lstm1, lstm2], axis=-1)
denseA = k.layers.Dense(30, name="dense1", activation=k.activations.sigmoid)(merged)
denseB = k.layers.Dense(4, name="dense2", activation=k.activations.softmax)(denseA)

model = k.models.Model(input_tensor, denseB)

#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata= tf.RunMetadata()

model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.categorical_crossentropy,
    metrics=[k.metrics.categorical_accuracy], 
    #options=run_options, 
    #run_metadata=run_metadata
)

print(model.summary())

modelSaver = k.callbacks.ModelCheckpoint(
    tfDataDir + f"{modelName}_checkpoint.h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True
)

history = model.fit_generator(
    training_generator,
    epochs=nrEpochs,
    verbose=1,
    callbacks=[modelSaver],
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=4,
)


# from tensorflow.python.client import timeline
# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()
# with open('timeline.json', 'w') as f:
#     f.write(ctf)

resultDir = f"{tfDataDir}/{modelName}_{int(time.time())}"
os.makedirs(resultDir)

model.save(f"{tfDataDir}/latestRadPredModel.h5")
model.save(f"{resultDir}/{modelName}.h5")
np.savez(f"{resultDir}/history.npz", history.history)

fig, ax = p.plotMultiPlot([history.history['loss'], history.history['val_loss']],
                          "epoch", "loss", "loss",
                          ["training", "validation"])
fig.savefig(f"{resultDir}/loss.png")

fig, ax = p.plotMultiPlot([history.history['categorical_accuracy'], history.history['val_categorical_accuracy']],
                          "epoch", "acuracy", "acuracy",
                          ["training", "validation"])
fig.savefig(f"{resultDir}/accuracy.png")

fig, ax = p.plotValidationHitGrid(model, validation_generator, np.min(100, batchSize))
fig.savefig(f"{resultDir}/hitGrid.png")

# with open(f"{resultDir}/{modelName}_description.txt", "w") as f:
#     model.summary()
