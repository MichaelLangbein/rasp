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



for X, y in validation_generator:
    yPred = model.predict(X)
    for sample in range(validation_generator.batchSize):
        fig, axes = plt.subplots(3)
        axes[0].imshow(y[sample, :, :, 0])
        axes[0].set_title("target")
        axes[1].imshow(yPred[sample, :, :, 0])
        axes[1].set_title("nowcast")
        animation = p.movie(fig, axes[2], X[sample, :, :, :, 0], [])
        axes[2].set_title("X")
        plt.show()
    break