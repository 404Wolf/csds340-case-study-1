#!/usr/bin/env python3
"""
Script to compare feature dropping between Random Forest importance and L1 regularization
and plot the percentage of features in common that are dropped by both methods.

@author: GitHub Copilot
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import sys
import os

# Add parent directory to path to import classifySpam
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifySpam import get_l1_preserved_features


def get_rf_dropped_features(trainFeatures, trainLabels, drop_percentage):
    """
    Get features dropped by Random Forest importance ranking.

    Returns:
        dropped_indices: Set of feature indices that would be dropped
        all_sorted_indices: All feature indices sorted by importance (least important first)
    """
    # Use Random Forest to get feature importance
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf.fit(trainFeatures, trainLabels)

    # Get feature importance and sort indices by importance (ascending)
    importance = rf.feature_importances_
    sorted_indices = np.argsort(importance)  # least important first

    # Calculate how many features to drop
    total_features = trainFeatures.shape[1]
    n_drop = int(total_features * drop_percentage / 100)

    # Get indices of features to drop (least important ones)
    dropped_indices = set(sorted_indices[:n_drop])

    return dropped_indices, sorted_indices, importance


def get_l1_dropped_features(trainFeatures, trainLabels, C):
    """
    Get features dropped by L1 regularization.

    Returns:
        dropped_indices: Set of feature indices that would be dropped
        preserved_indices: List of preserved feature indices
    """
    preserved_indices = get_l1_preserved_features(trainFeatures, trainLabels, C)
    total_features = trainFeatures.shape[1]

    # Convert to sets for easier comparison
    preserved_set = set(preserved_indices)
    all_features = set(range(total_features))
    dropped_indices = all_features - preserved_set

    return dropped_indices, preserved_indices


def analyze_feature_overlap(trainFeatures, trainLabels, drop_percentages, c_values):
    """
    Analyze the overlap between features dropped by RF importance and L1 regularization.

    Returns:
        overlap_data: Dictionary containing overlap analysis results
    """
    total_features = trainFeatures.shape[1]

    # Preprocess data
    imputer = SimpleImputer(missing_values=-1, strategy="median")
    train_imputed = imputer.fit_transform(trainFeatures)

    overlap_data = {
        "drop_percentages": [],
        "c_values": [],
        "rf_dropped_counts": [],
        "l1_dropped_counts": [],
        "common_dropped_counts": [],
        "overlap_percentages": [],
        "jaccard_similarities": [],
    }

    print("Analyzing feature overlap between Random Forest and L1 regularization...")

    for drop_pct in drop_percentages:
        print(f"\nAnalyzing {drop_pct}% feature dropping...")

        # Get RF dropped features
        rf_dropped, rf_sorted, rf_importance = get_rf_dropped_features(
            train_imputed, trainLabels, drop_pct
        )
        n_rf_dropped = len(rf_dropped)

        if n_rf_dropped == 0:
            continue

        # Find C value that gives similar number of dropped features
        best_c = None
        best_diff = float("inf")

        for C in c_values:
            try:
                l1_dropped, l1_preserved = get_l1_dropped_features(
                    train_imputed, trainLabels, C
                )
                n_l1_dropped = len(l1_dropped)
                diff = abs(n_l1_dropped - n_rf_dropped)

                if diff < best_diff:
                    best_diff = diff
                    best_c = C
                    best_l1_dropped = l1_dropped
                    best_n_l1_dropped = n_l1_dropped

            except Exception as e:
                continue

        if best_c is None:
            continue

        # Calculate overlap
        common_dropped = rf_dropped.intersection(best_l1_dropped)
        n_common = len(common_dropped)

        # Calculate overlap percentage (of the smaller set)
        min_dropped = min(n_rf_dropped, best_n_l1_dropped)
        overlap_pct = (n_common / min_dropped * 100) if min_dropped > 0 else 0

        # Calculate Jaccard similarity (intersection over union)
        union_dropped = rf_dropped.union(best_l1_dropped)
        jaccard = (n_common / len(union_dropped)) if len(union_dropped) > 0 else 0

        # Store results
        overlap_data["drop_percentages"].append(drop_pct)
        overlap_data["c_values"].append(best_c)
        overlap_data["rf_dropped_counts"].append(n_rf_dropped)
        overlap_data["l1_dropped_counts"].append(best_n_l1_dropped)
        overlap_data["common_dropped_counts"].append(n_common)
        overlap_data["overlap_percentages"].append(overlap_pct)
        overlap_data["jaccard_similarities"].append(jaccard)

        print(
            f"  RF dropped: {n_rf_dropped}, L1 dropped: {best_n_l1_dropped} (C={best_c:.3f})"
        )
        print(
            f"  Common dropped: {n_common}, Overlap: {overlap_pct:.1f}%, Jaccard: {jaccard:.3f}"
        )

    return overlap_data


def plot_feature_overlap(overlap_data):
    """
    Create comprehensive plots showing feature overlap analysis.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    drop_percentages = overlap_data["drop_percentages"]

    # Plot 1: Number of features dropped by each method
    ax1.plot(
        drop_percentages,
        overlap_data["rf_dropped_counts"],
        "o-",
        label="Random Forest",
        color="tab:blue",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        drop_percentages,
        overlap_data["l1_dropped_counts"],
        "s-",
        label="L1 Regularization",
        color="tab:red",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        drop_percentages,
        overlap_data["common_dropped_counts"],
        "^-",
        label="Common Dropped",
        color="tab:green",
        linewidth=2,
        markersize=6,
    )

    ax1.set_xlabel("Target Drop Percentage (%)")
    ax1.set_ylabel("Number of Features Dropped")
    ax1.set_title("Features Dropped by Each Method")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Overlap percentage
    ax2.plot(
        drop_percentages,
        overlap_data["overlap_percentages"],
        "o-",
        color="tab:purple",
        linewidth=2,
        markersize=6,
    )
    ax2.set_xlabel("Target Drop Percentage (%)")
    ax2.set_ylabel("Overlap Percentage (%)")
    ax2.set_title("Percentage of Common Features Dropped\n(Relative to Smaller Set)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Plot 3: Jaccard similarity
    ax3.plot(
        drop_percentages,
        overlap_data["jaccard_similarities"],
        "o-",
        color="tab:orange",
        linewidth=2,
        markersize=6,
    )
    ax3.set_xlabel("Target Drop Percentage (%)")
    ax3.set_ylabel("Jaccard Similarity")
    ax3.set_title("Jaccard Similarity of Dropped Features\n(Intersection over Union)")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Plot 4: C values used for matching
    ax4.semilogx(
        drop_percentages,
        overlap_data["c_values"],
        "o-",
        color="tab:brown",
        linewidth=2,
        markersize=6,
    )
    ax4.set_xlabel("Target Drop Percentage (%)")
    ax4.set_ylabel("C Value (L1 Regularization)")
    ax4.set_title("C Values Used to Match Feature Drop Count")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_overlap_superposition(overlap_data):
    """
    Create a superposition plot showing both overlap metrics on the same axes.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    drop_percentages = overlap_data["drop_percentages"]

    # Plot overlap percentage on primary y-axis
    color1 = "tab:purple"
    ax1.set_xlabel("Target Drop Percentage (%)")
    ax1.set_ylabel("Overlap Percentage (%)", color=color1)
    line1 = ax1.plot(
        drop_percentages,
        overlap_data["overlap_percentages"],
        "o-",
        color=color1,
        label="Overlap Percentage",
        linewidth=3,
        markersize=8,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis for Jaccard similarity
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Jaccard Similarity", color=color2)
    line2 = ax2.plot(
        drop_percentages,
        overlap_data["jaccard_similarities"],
        "s-",
        color=color2,
        label="Jaccard Similarity",
        linewidth=3,
        markersize=8,
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    # Add title and combined legend
    plt.title(
        "Feature Dropping Agreement: Random Forest vs L1 Regularization\\n"
        + "Superposition of Overlap Metrics",
        fontsize=16,
        pad=20,
    )

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=12)

    # Add text annotation
    ax1.text(
        0.02,
        0.98,
        "Higher values indicate better agreement\\nbetween feature selection methods",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def main():
    """Main function to run the feature overlap analysis."""
    # Load data
    trainDataFilename = "../spamTrain1.csv"

    print("Loading data...")
    trainData = np.loadtxt(trainDataFilename, delimiter=",")

    # Separate labels (last column) from training data
    trainFeatures = trainData[:, :-1]
    trainLabels = trainData[:, -1]

    print(
        f"Training set: {trainFeatures.shape[0]} samples, {trainFeatures.shape[1]} features"
    )

    # Define analysis parameters
    drop_percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    c_values = np.logspace(-3, 3, 100)  # C values from 0.001 to 1000

    print(f"Testing drop percentages: {drop_percentages}")
    print(f"C value range: {c_values.min():.3f} to {c_values.max():.1f}")

    # Analyze feature overlap
    overlap_data = analyze_feature_overlap(
        trainFeatures, trainLabels, drop_percentages, c_values
    )

    if not overlap_data["drop_percentages"]:
        print("No valid overlap data generated. Exiting.")
        return

    # Create plots
    print("\\nCreating plots...")

    # Comprehensive analysis plot
    fig1 = plot_feature_overlap(overlap_data)
    plt.savefig("feature_overlap_analysis.png", dpi=300, bbox_inches="tight")
    print("Comprehensive overlap analysis saved as 'feature_overlap_analysis.png'")

    # Superposition plot
    fig2 = plot_overlap_superposition(overlap_data)
    plt.savefig("feature_overlap_superposition.png", dpi=300, bbox_inches="tight")
    print("Superposition plot saved as 'feature_overlap_superposition.png'")

    # Show plots
    plt.show()

    # Print summary statistics
    print("\\nSummary Statistics:")
    print(
        f"Average overlap percentage: {np.mean(overlap_data['overlap_percentages']):.1f}%"
    )
    print(
        f"Maximum overlap percentage: {np.max(overlap_data['overlap_percentages']):.1f}%"
    )
    print(
        f"Minimum overlap percentage: {np.min(overlap_data['overlap_percentages']):.1f}%"
    )
    print(
        f"Average Jaccard similarity: {np.mean(overlap_data['jaccard_similarities']):.3f}"
    )
    print(
        f"Maximum Jaccard similarity: {np.max(overlap_data['jaccard_similarities']):.3f}"
    )

    # Detailed results table
    print(f"\\nDetailed Results:")
    print(
        f"{'Drop%':<6} {'RF_Drop':<8} {'L1_Drop':<8} {'Common':<7} {'Overlap%':<8} {'Jaccard':<8} {'C_Value':<10}"
    )
    print("-" * 65)
    for i in range(len(overlap_data["drop_percentages"])):
        print(
            f"{overlap_data['drop_percentages'][i]:<6} "
            f"{overlap_data['rf_dropped_counts'][i]:<8} "
            f"{overlap_data['l1_dropped_counts'][i]:<8} "
            f"{overlap_data['common_dropped_counts'][i]:<7} "
            f"{overlap_data['overlap_percentages'][i]:<8.1f} "
            f"{overlap_data['jaccard_similarities'][i]:<8.3f} "
            f"{overlap_data['c_values'][i]:<10.3f}"
        )


if __name__ == "__main__":
    main()
