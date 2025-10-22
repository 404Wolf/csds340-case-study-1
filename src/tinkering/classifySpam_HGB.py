# -*- coding: utf-8 -*-
"""
Histogram Gradient Boosting Classifier with Configurable Preprocessing
@author: Best models suite
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

# ==================== CONFIGURATION ====================
USE_FEATURE_SELECTION = False  # Set to True to enable RF-based feature selection
IMPUTATION_STRATEGY = "median"  # Options: 'mean', 'median', 'knn', 'iterative'
SCALING_METHOD = "none"  # Options: 'none', 'standard', 'robust', 'minmax'
# ======================================================


def _create_pipeline():
    """Create HGB pipeline with configurable preprocessing"""
    steps = []

    # 1. Imputation (handles missing values coded as -1)
    if IMPUTATION_STRATEGY == "mean":
        steps.append(("impute", SimpleImputer(missing_values=-1, strategy="mean")))
    elif IMPUTATION_STRATEGY == "median":
        steps.append(("impute", SimpleImputer(missing_values=-1, strategy="median")))
    elif IMPUTATION_STRATEGY == "knn":
        # Replace -1 with NaN first for KNN imputer
        steps.append(("impute", KNNImputer(missing_values=-1, n_neighbors=5)))
    elif IMPUTATION_STRATEGY == "iterative":
        steps.append(("impute", IterativeImputer(missing_values=-1, random_state=42)))

    # 2. Remove constant features
    steps.append(("nzvar", VarianceThreshold(threshold=0.0)))

    # 3. Scaling (optional)
    if SCALING_METHOD == "standard":
        steps.append(("scale", StandardScaler()))
    elif SCALING_METHOD == "robust":
        steps.append(("scale", RobustScaler()))
    elif SCALING_METHOD == "minmax":
        steps.append(("scale", MinMaxScaler()))

    # 4. Feature Selection (optional)
    if USE_FEATURE_SELECTION:
        steps.append(
            (
                "feature_selection",
                SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1
                    ),
                    threshold="median",
                ),
            )
        )

    # 5. Model with regularization to reduce overfitting
    steps.append(
        (
            "model",
            HistGradientBoostingClassifier(
                random_state=42,
                learning_rate=0.05,
                max_depth=5,
                min_samples_leaf=50,
                max_iter=200,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        )
    )

    return Pipeline(steps)


def aucCV(features, labels):
    """10-fold stratified cross-validation"""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = _create_pipeline()
    scores = cross_val_score(
        model, features, labels, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return scores


def predictTest(trainFeatures, trainLabels, testFeatures):
    """Train and predict probabilities"""
    model = _create_pipeline()
    model.fit(trainFeatures, trainLabels)
    return model.predict_proba(testFeatures)[:, 1]


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "src", "spamTrain1.csv")
    data = np.loadtxt(data_path, delimiter=",")

    features = data[:, :-1]
    labels = data[:, -1]

    print("=" * 60)
    print("HISTOGRAM GRADIENT BOOSTING CLASSIFIER")
    print("=" * 60)
    print(f"Feature Selection: {USE_FEATURE_SELECTION}")
    print(f"Imputation: {IMPUTATION_STRATEGY}")
    print(f"Scaling: {SCALING_METHOD}")
    print()

    scores = aucCV(features, labels)
    print(f"10-fold CV mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Train/test split (odd/even)
    trainFeatures = features[0::2, :]
    trainLabels = labels[0::2]
    testFeatures = features[1::2, :]
    testLabels = labels[1::2]

    testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
    print(f"Test set AUC: {roc_auc_score(testLabels, testOutputs):.4f}")
