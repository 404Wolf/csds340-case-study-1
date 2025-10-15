# -*- coding: utf-8 -*-
"""
Script used to evaluate classifier accuracy

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from classifySpam import predictTest

desiredFPR = 0.01
dataFilename = "spamTrain1.csv"


def tprAtFPR(labels, outputs, desiredFPR):
    fpr, tpr, thres = roc_curve(labels, outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = (tprAbove - tprBelow) / (fprAbove - fprBelow) * (
        desiredFPR - fprBelow
    ) + tprBelow
    return tprAt, fpr, tpr


# Load data and split into train/test using odd/even split
data = np.loadtxt(dataFilename, delimiter=",")
features = data[:, :-1]
labels = data[:, -1]

# Arbitrarily choose all odd samples as train set and all even as test set
trainFeatures = features[0::2, :]
trainLabels = labels[0::2]
testFeatures = features[1::2, :]
testLabels = labels[1::2]

testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
aucTestRun = roc_auc_score(testLabels, testOutputs)
tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels, testOutputs, desiredFPR)

plt.plot(fpr, tpr)

print(f"Test set AUC: {aucTestRun}")
print(f"TPR at FPR = {desiredFPR}: {tprAtDesiredFPR}")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve for spam detector")
plt.show()
