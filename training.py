import os
import sys
import time
import datetime as dt
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from data import Film, Frame, analyseAndSaveTimeRange, DataGenerator, frameHeight, frameWidth, frameLength, getFileNames
from config import rawDataDir, processedDataDir, tfDataDir, trainingStart, trainingEnd, validationStart, validationEnd
import plotting as p
import matplotlib.pyplot as plt


modelName = "Nowcast"

batchSize = 20
nrBatchesPerEpoch = 50
nrEpochs = 12
timeSteps = int(5 * 60 / 5)
timeSeriesOffset = 5
channels = 1


fileNames = getFileNames(processedDataDir, trainingStart, validationEnd)
trainingFilenames = []
validationFilenames = []
for i, filename in enumerate(fileNames):
    if i%4 == 0:
        validationFilenames.append(filename)
    else:
        trainingFilenames.append(filename)
training_generator = DataGenerator(trainingFilenames, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)
validation_generator = DataGenerator(validationFilenames, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)



input_tensor = k.Input(shape=(timeSteps, frameHeight, frameWidth, channels))
convLstm = k.layers.ConvLSTM2D(1, kernel_size=(2, 2), activation="relu", padding="same", data_format="channels_last")(input_tensor)
model = k.models.Model(input_tensor, convLstm)

#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata= tf.RunMetadata()

model.compile(
    optimizer=k.optimizers.Adam(lr=0.01),
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



maxPlots = 5
plot = 0
for X, y in validation_generator:
    yPred = model.predict(X)
    for sample in range(validation_generator.batchSize):
        fig, axes = plt.subplots(2, 3)

        img1 = axes[0, 0].imshow(y[sample, :, :, 0])
        img1.norm.vmin = np.min(X)
        img1.norm.vmax = np.max(X)
        axes[0, 0].set_title("target")

        img2 = axes[0, 1].imshow(yPred[sample, :, :, 0])
        img2.norm.vmin = np.min(X)
        img2.norm.vmax = np.max(X)
        axes[0, 1].set_title("nowcast")

        img3 = axes[0, 2].imshow(np.abs( yPred[sample, :, :, 0] - y[sample, :, :, 0] ))
        img3.norm.vmin = np.min(X)
        img3.norm.vmax = np.max(X)
        axes[0, 2].set_title("difference")

        B, T, H, W, C = X.shape
        movieDataList = []
        for t in range(T):
            frameData = X[sample, t, :, :, 0]
            if np.max(frameData) >= 0.1 * np.max(X):
                movieDataList.append(frameData)
        movieDataList.append(yPred[sample, :, :, 0])
        movieData = np.array(movieDataList)
        animation1 = p.movie(fig, axes[1, 1], movieData, [], interval=300, repeat=True, repeat_delay=1000)
        axes[1, 1].set_title("Predicted movie")

        movieData2 = np.copy(movieData)
        movieData2[-1] = y[sample, :, :, 0]
        animation2 = p.movie(fig, axes[1, 0], movieData2, [], interval=300, repeat=True, repeat_delay=1000)
        axes[1, 0].set_title("Real movie")


        plt.show()
        plot += 1
        if plot >= maxPlots:
            break
    if plot >= maxPlots:
        break