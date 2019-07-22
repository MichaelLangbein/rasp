import os
import datetime as dt
from config import rawDataDir, processedDataDir, tfDataDir
import data as rd
from data import rawDataDir, processedDataDir, frameHeight, frameWidth, frameLength, Film, Frame, analyseAndSaveTimeRange
import numpy as np
import tensorflow.keras as k
import matplotlib.pyplot as plt
import plotting as pl


print("----starting up-----")
modelPath = resultDir = f"{tfDataDir}/" + "/SimpleConvolutional3D_1563548170/SimpleConvolutional3D.h5"
maxBatchesPerEpoch = 100
batchSize = 4
timeSteps = int(5 * 60 / 5)



model = k.models.load_model(modelPath)
generator = rd.DataGenerator(processedDataDir, dt.datetime(2016, 7, 1), dt.datetime(2016, 7, 15), maxBatchesPerEpoch, batchSize, timeSteps, 5, True)




def getMaxIndex(list):
    return np.where(list == np.amax(list))


print("----predicting----")
maxSamples = 1000
i = 0
results = []
for dataIn, dataOut in generator:

    predictions = model.predict(dataIn)

    # maxActInpt = pl.getMaximallyActivatingImage(model, "conv4", 0, dataIn[0].shape)
    # pl.showMovie(maxActInpt[:, :, :, 0], ["maximal activationImage for conv4 channel 0 "], 100)

    for r in range(len(predictions)):
        target = dataOut[r]
        maxIndexTarget = getMaxIndex(target)
        prediction = predictions[r]
        maxIndexPrediction = getMaxIndex(prediction)
        print(f"{i}  prediction: {prediction}    target: {target}   correctly predicted {maxIndexPrediction == maxIndexTarget}")
        # pl.showActivation(model, dataIn[r], "conv3", 2)
        results.append({"prediction": prediction, "target": target})
        i += 1
        if i > maxSamples:
            break
    if i > maxSamples:
        break





preds = np.zeros((4, 4))
for result in results:
    targetCat = getMaxIndex(result["target"])
    predicCat = getMaxIndex(result["prediction"])
    preds[targetCat, predicCat] += 1



categories = ["keine", "starkregen", "heftiger", "extremer"]
fig, ax = plt.subplots()
im = ax.imshow(preds)
ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(4))
ax.set_xticklabels(categories)
ax.set_yticklabels(categories)
ax.set_ylabel("target")
ax.set_xlabel("predicted")
for i in range(4):
    for j in range(4):
        text = ax.text(j, i, preds[i, j], ha="center", va="center", color="w")
fig.tight_layout()
plt.show()
