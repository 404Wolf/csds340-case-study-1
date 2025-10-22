#!/usr/bin/env python3
"""
Script to plot AUC and number of dropped features vs C parameter
for L1-regularized logistic regression feature selection

@author: GitHub Copilot
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from classifySpam import predictTest, get_l1_preserved_features
from sklearn.impute import SimpleImputer


def analyze_c_values(trainFeatures, trainLabels, testFeatures, testLabels, c_values):
    """
    Analyze the effect of different C values on AUC and feature selection.

    Returns:
        auc_scores: List of AUC scores for each C value
        features_kept: List of number of features kept for each C value
        features_dropped: List of number of features dropped for each C value
    """
    auc_scores = []
    features_kept = []
    features_dropped = []
    total_features = trainFeatures.shape[1]

    # Preprocess data once
    imputer = SimpleImputer(missing_values=-1, strategy="median")
    train_imputed = imputer.fit_transform(trainFeatures)

    for C in c_values:
        print(f"Evaluating C = {C}")

        try:
            # Get predictions with this C value
            testOutputs = predictTest(trainFeatures, trainLabels, testFeatures, C=C)

            # Calculate AUC
            auc = roc_auc_score(testLabels, testOutputs)
            auc_scores.append(auc)

            # Get number of preserved features
            preserved_idx = get_l1_preserved_features(train_imputed, trainLabels, C=C)
            n_kept = len(preserved_idx)

            # If no features are preserved, all features are effectively kept (fallback)
            if n_kept == 0:
                n_kept = total_features

            n_dropped = total_features - n_kept

            features_kept.append(n_kept)
            features_dropped.append(n_dropped)

            print(
                f"  AUC: {auc:.4f}, Features kept: {n_kept}/{total_features}, Dropped: {n_dropped}"
            )

        except Exception as e:
            print(f"  Error with C={C}: {e}")
            # Skip this C value
            continue

    return auc_scores, features_kept, features_dropped


def plot_c_analysis(c_values, auc_scores, features_dropped):
    """
    Plot AUC and number of dropped features vs C values.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot AUC on primary y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("C (Regularization Parameter)")
    ax1.set_ylabel("AUC Score", color=color1)
    line1 = ax1.plot(
        c_values,
        auc_scores,
        "o-",
        color=color1,
        label="AUC Score",
        linewidth=2,
        markersize=6,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")  # Log scale for C values
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis for number of dropped features
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Number of Dropped Features", color=color2)
    line2 = ax2.plot(
        c_values,
        features_dropped,
        "s-",
        color=color2,
        label="Features Dropped",
        linewidth=2,
        markersize=6,
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Add title and legend
    plt.title(
        "AUC Score and Feature Dropping vs Regularization Parameter C",
        fontsize=14,
        pad=20,
    )

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")

    # Improve layout
    fig.tight_layout()

    return fig


def main():
    """Main function to run the C value analysis."""
    # Load data
    trainDataFilename = "spamTrain1.csv"
    testDataFilename = "spamTrain2.csv"

    print("Loading data...")
    trainData = np.loadtxt(trainDataFilename, delimiter=",")
    testData = np.loadtxt(testDataFilename, delimiter=",")

    # Separate labels (last column) from training and test data
    trainFeatures = trainData[:, :-1]
    trainLabels = trainData[:, -1]
    testFeatures = testData[:, :-1]
    testLabels = testData[:, -1]

    print(
        f"Training set: {trainFeatures.shape[0]} samples, {trainFeatures.shape[1]} features"
    )
    print(
        f"Test set: {testFeatures.shape[0]} samples, {testFeatures.shape[1]} features"
    )

    # Define range of C values to test (logarithmic scale with more diversity)
    c_values = np.linspace(1, 80, 100)

    print(f"\nTesting C values: {c_values}")

    # Analyze different C values
    auc_scores, features_kept, features_dropped = analyze_c_values(
        trainFeatures, trainLabels, testFeatures, testLabels, c_values
    )

    # Create and show plot
    print("\nCreating plot...")
    fig = plot_c_analysis(c_values, auc_scores, features_dropped)

    # Save the plot
    plt.savefig("c_analysis_plot.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'c_analysis_plot.png'")

    # Show the plot
    plt.show()

    # Print summary
    print(f"\nSummary:")
    print(f"Best AUC: {max(auc_scores):.4f} at C = {c_values[np.argmax(auc_scores)]}")
    print(
        f"Minimum features dropped: {min(features_dropped)} at C = {c_values[np.argmin(features_dropped)]}"
    )
    print(
        f"Maximum features dropped: {max(features_dropped)} at C = {c_values[np.argmax(features_dropped)]}"
    )

    # Create a summary table
    print(f"\nDetailed Results:")
    print(f"{'C':<8} {'AUC':<8} {'Kept':<6} {'Dropped':<8}")
    print("-" * 32)
    for i, c in enumerate(c_values):
        print(
            f"{c:<8} {auc_scores[i]:<8.4f} {features_kept[i]:<6} {features_dropped[i]:<8}"
        )


if __name__ == "__main__":
    main()
