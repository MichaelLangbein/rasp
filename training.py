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
import matplotlib.pyplot as plt


modelName = "Nowcast"

batchSize = 20
nrBatchesPerEpoch = 30
nrEpochs = 4
timeSteps = int(5 * 60 / 5)
timeSeriesOffset = 5
channels = 1


training_generator = DataGenerator(processedDataDir, trainingStart, trainingEnd, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)
validation_generator = DataGenerator(processedDataDir, validationStart, validationEnd, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)


input_tensor = k.Input(shape=(timeSteps, frameHeight, frameWidth, channels))
convLstm = k.layers.ConvLSTM2D(1, kernel_size=(2, 2), padding="same", data_format="channels_last")(input_tensor)
model = k.models.Model(input_tensor, convLstm)

#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata= tf.RunMetadata()

model.compile(
    optimizer=k.optimizers.Adam(),
    loss=k.losses.mse,
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


for X, y in validation_generator:
    yPred = model.predict(X)
    for sample in range(validation_generator.batchSize):
        fig, axes = plt.subplots(3)
        axes[0].imshow(y[sample])
        axes[0].set_title("target")
        axes[1].imshow(yPred[sample])
        axes[1].set_title("nowcast")
        animation = p.movie(fig, axes[2], X, [])
        axes[2].set_title("X")
        plt.show()
    break