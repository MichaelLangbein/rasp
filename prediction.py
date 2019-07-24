import os
import datetime as dt
import data as rd
from config import rawDataDir, processedDataDir, tfDataDir, validationStart, validationEnd
from data import frameHeight, frameWidth, frameLength, Film, Frame, analyseAndSaveTimeRange, DataGenerator
import numpy as np
import tensorflow.keras as k
import matplotlib.pyplot as plt
import plotting as p


print("----starting up-----")
resultDir = f"{tfDataDir}/Nowcast_1563956451"
modelPath = f"{resultDir}/Nowcast.h5"
maxBatchesPerEpoch = 100
batchSize = 4
timeSteps = int(5 * 60 / 5)
nrBatchesPerEpoch = 30
timeSeriesOffset = 5
channels = 1



model = k.models.load_model(modelPath)
validation_generator = DataGenerator(processedDataDir, validationStart, validationEnd, batchSize, timeSteps, timeSeriesOffset, nrBatchesPerEpoch)


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