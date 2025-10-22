# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Gaussian naive Bayes on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def create_model():
    return make_pipeline(
        SimpleImputer(missing_values=-1, strategy="mean"), GaussianNB()
    )


def aucCV(features, labels):
    model = create_model()
    scores = cross_val_score(model, features, labels, cv=10, scoring="roc_auc")

    return scores


def create_logistic_model():
    return make_pipeline(
        SimpleImputer(missing_values=-1, strategy="mean"), LogisticRegression(random_state=42)
    )


def logistic_regression_predict(trainFeatures, trainLabels, testFeatures):
    """
    Train a logistic regression model and return class predictions (0 or 1) for test data.
    
    Args:
        trainFeatures: Training feature matrix
        trainLabels: Training labels
        testFeatures: Test feature matrix to predict
        
    Returns:
        testPredictions: Array of predicted class labels (0 or 1)
    """
    model = create_logistic_model()
    model.fit(trainFeatures, trainLabels)
    
    # Get class predictions (0 or 1)
    testPredictions = model.predict(testFeatures)
    
    return testPredictions


def logistic_regression_probabilities(trainFeatures, trainLabels, testFeatures):
    """
    Train a logistic regression model and return class probabilities for test data.
    
    Args:
        trainFeatures: Training feature matrix
        trainLabels: Training labels
        testFeatures: Test feature matrix to predict
        
    Returns:
        testProbabilities: Array of predicted probabilities for positive class
    """
    model = create_logistic_model()
    model.fit(trainFeatures, trainLabels)
    
    # Get probability predictions for positive class (class 1)
    testProbabilities = model.predict_proba(testFeatures)[:, 1]
    
    return testProbabilities


def predictTest(trainFeatures, trainLabels, testFeatures):
    model = create_model()
    model.fit(trainFeatures, trainLabels)

    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:, 1]

    return testOutputs


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt("spamTrain1.csv", delimiter=",")
    # Separate labels (last column)
    features = data[:, :-1]
    labels = data[:, -1]

    # Evaluating classifier accuracy using 10-fold cross-validation
    print("10-fold cross-validation mean AUC: ", np.mean(aucCV(features, labels)))

    # Arbitrarily choose all odd samples as train set and all even as test set
    # then compute test set AUC for model trained only on fixed train set
    trainFeatures = features[0::2, :]
    trainLabels = labels[0::2]
    testFeatures = features[1::2, :]
    testLabels = labels[1::2]
    testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels, testOutputs))

    # Logistic regression predictions
    logistic_predictions = logistic_regression_predict(trainFeatures, trainLabels, testFeatures)
    logistic_probabilities = logistic_regression_probabilities(trainFeatures, trainLabels, testFeatures)
    
    print("Logistic regression class predictions:", logistic_predictions)
    print("Logistic regression AUC:", roc_auc_score(testLabels, logistic_probabilities))
    print("Logistic regression accuracy:", np.mean(logistic_predictions == testLabels))

    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(nTestExamples), testLabels[sortIndex], "b.")
    plt.xlabel("Sorted example number")
    plt.ylabel("Target")
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(nTestExamples), testOutputs[sortIndex], "r.")
    plt.xlabel("Sorted example number")
    plt.ylabel("Output (predicted target)")
    plt.tight_layout()
    plt.show()
