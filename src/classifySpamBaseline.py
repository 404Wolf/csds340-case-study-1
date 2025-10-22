# -*- coding: utf-8 -*-
"""
Optimized spam classifier - ExtraTrees with top 16 features
Based on incremental feature analysis showing peak TPR@FPR=0.01 at 16 features

@author: Kevin S. Xu (modified)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

# Top 16 features by importance (peak TPR configuration)
# Order: 19, 8, 3, 29, 4, 7, 6, 12, 28, 0, 17, 10, 9, 13, 18, 23

#BEST_FEATURES = [19, 8, 3, 29, 4, 7, 6, 12, 28, 0, 17, 10, 9, 13, 18, 23]
BEST_FEATURES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29]


def preprocess_features(features):
    """Simple preprocessing: impute missing values"""
    imputer = SimpleImputer(missing_values=-1, strategy="median")
    return imputer.fit_transform(features)


def aucCV(features, labels):
    """
    10-fold cross-validation with optimized ExtraTrees
    Uses top 16 features for best TPR@FPR=0.01
    """
    # Select best features
    features_selected = features[:, BEST_FEATURES]
    
    # Preprocess
    features_processed = preprocess_features(features_selected)
    
    # Model with optimized parameters
    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features='sqrt',
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion='entropy',
        class_weight='balanced',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, features_processed, labels, 
                            cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores


def predictTest(trainFeatures, trainLabels, testFeatures):
    """
    Train model and predict on test set
    Uses top 16 features for best TPR@FPR=0.01
    """
    # Select best features
    train_selected = trainFeatures[:, BEST_FEATURES]
    test_selected = testFeatures[:, BEST_FEATURES]
    
    # Preprocess
    imputer = SimpleImputer(missing_values=-1, strategy="median")
    train_processed = imputer.fit_transform(train_selected)
    test_processed = imputer.transform(test_selected)
    
    # Train model
    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features='sqrt',
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion='entropy',
        class_weight='balanced',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(train_processed, trainLabels)
    
    # Return probability of positive class
    return model.predict_proba(test_processed)[:, 1]


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    data = np.loadtxt("spamTrain1.csv", delimiter=",")
    features = data[:, :-1]
    labels = data[:, -1]

    print("\n" + "="*70)
    print("OPTIMIZED EXTRATREES - TOP 16 FEATURES (PEAK TPR)")
    print("="*70)
    print(f"Total features: {features.shape[1]}")
    print(f"Using top 16 features: {BEST_FEATURES}")
    print(f"Expected TPR@FPR=0.01: 0.7472")
    print(f"Expected CV AUC: 0.9075")
    
    # Cross-validation
    print("\n" + "="*70)
    print("10-FOLD CROSS-VALIDATION")
    print("="*70)
    cv_scores = aucCV(features, labels)
    print(f"CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"Min: {np.min(cv_scores):.4f}, Max: {np.max(cv_scores):.4f}")
    
    # Test set evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION (Odd/Even Split)")
    print("="*70)
    trainFeatures = features[0::2, :]
    trainLabels = labels[0::2]
    testFeatures = features[1::2, :]
    testLabels = labels[1::2]
    
    testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
    test_auc = roc_auc_score(testLabels, testOutputs)
    print(f"Test AUC: {test_auc:.4f}")
    print("="*70)

    # Visualize predictions
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(nTestExamples), testLabels[sortIndex], "b.", label="True Labels")
    plt.xlabel("Sorted example number", fontsize=11)
    plt.ylabel("Label", fontsize=11)
    plt.title("True Labels (Sorted)", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(nTestExamples), testOutputs[sortIndex], "r.", label="Predictions")
    plt.xlabel("Sorted example number", fontsize=11)
    plt.ylabel("Predicted Probability", fontsize=11)
    plt.title("Model Predictions (Spam Probability)", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
