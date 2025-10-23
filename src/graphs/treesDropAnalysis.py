import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer


def get_rf_feature_importance(trainFeatures, trainLabels):
    """
    Get feature importance using Random Forest.
    Returns indices sorted by importance (ascending order - least important first).
    """
    # Use a simple Random Forest to get feature importance
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf.fit(trainFeatures, trainLabels)

    # Get feature importance and sort indices by importance (ascending)
    importance = rf.feature_importances_
    sorted_indices = np.argsort(importance)  # least important first

    return sorted_indices, importance


def getModelWithFeatureSubset(
    trainFeatures, trainLabels, testFeatures, feature_indices
):
    """
    Train model using only specified feature indices.
    """
    # Select features
    train_selected = trainFeatures[:, feature_indices]
    test_selected = testFeatures[:, feature_indices]

    # Train ExtraTreesClassifier on selected features
    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        criterion="entropy",
        class_weight="balanced",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train_selected, trainLabels)

    return model, test_selected


def analyze_feature_dropping(
    trainFeatures, trainLabels, testFeatures, testLabels, drop_percentages
):
    """
    Analyze the effect of dropping worst features based on Random Forest importance.

    Args:
        drop_percentages: List of percentages of worst features to drop (0-100)

    Returns:
        auc_scores: List of AUC scores for each drop percentage
        features_kept: List of number of features kept for each drop percentage
        features_dropped: List of number of features dropped for each drop percentage
    """
    auc_scores = []
    features_kept = []
    features_dropped = []
    total_features = trainFeatures.shape[1]

    # Preprocess data once
    imputer = SimpleImputer(missing_values=-1, strategy="median")
    train_imputed = imputer.fit_transform(trainFeatures)
    test_imputed = imputer.transform(testFeatures)

    print(f"Total features: {total_features}")

    # Get feature importance ranking
    print("Computing feature importance using Random Forest...")
    sorted_indices, importance = get_rf_feature_importance(train_imputed, trainLabels)

    for drop_pct in drop_percentages:
        print(f"\nEvaluating dropping {drop_pct}% of worst features")

        # Calculate how many features to drop
        n_drop = int(total_features * drop_pct / 100)
        n_keep = total_features - n_drop

        if n_keep <= 0:
            print(f"  Skipping: Would keep {n_keep} features")
            continue

        # Select features to keep (drop the least important ones)
        features_to_keep = sorted_indices[n_drop:]  # skip the n_drop least important

        try:
            # Train model with selected features
            model, test_selected = getModelWithFeatureSubset(
                train_imputed, trainLabels, test_imputed, features_to_keep
            )

            # Get predictions
            testOutputs = model.predict_proba(test_selected)[:, 1]

            # Calculate AUC
            auc = roc_auc_score(testLabels, testOutputs)
            auc_scores.append(auc)

            features_kept.append(n_keep)
            features_dropped.append(n_drop)

            print(
                f"  AUC: {auc:.4f}, Features kept: {n_keep}/{total_features}, Dropped: {n_drop}"
            )

        except Exception as e:
            print(f"  Error with {drop_pct}% drop: {e}")
            continue

    return auc_scores, features_kept, features_dropped, sorted_indices, importance


def plot_feature_drop_analysis(drop_percentages, auc_scores, features_dropped):
    """
    Plot AUC and number of dropped features vs drop percentage.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot AUC on primary y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("Percentage of Worst Features Dropped (%)")
    ax1.set_ylabel("AUC Score", color=color1)
    line1 = ax1.plot(
        drop_percentages[: len(auc_scores)],
        auc_scores,
        "o-",
        color=color1,
        label="AUC Score",
        linewidth=2,
        markersize=6,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Create secondary y-axis for number of dropped features
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Number of Dropped Features", color=color2)
    line2 = ax2.plot(
        drop_percentages[: len(features_dropped)],
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
        "AUC Score and Feature Dropping vs Percentage of Worst Features Dropped\n(Using Random Forest Feature Importance)",
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


def plot_feature_importance(importance, sorted_indices, top_n=20):
    """
    Plot feature importance for top and bottom features.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot top features
    top_indices = sorted_indices[-top_n:]  # most important
    top_importance = importance[top_indices]
    ax1.barh(range(top_n), top_importance)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([f"Feature {idx}" for idx in top_indices])
    ax1.set_xlabel("Feature Importance")
    ax1.set_title(f"Top {top_n} Most Important Features")
    ax1.grid(True, alpha=0.3)

    # Plot bottom features
    bottom_indices = sorted_indices[:top_n]  # least important
    bottom_importance = importance[bottom_indices]
    ax2.barh(range(top_n), bottom_importance)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([f"Feature {idx}" for idx in bottom_indices])
    ax2.set_xlabel("Feature Importance")
    ax2.set_title(f"Top {top_n} Least Important Features")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main function to run the feature dropping analysis."""
    # Load data
    trainDataFilename = "./spamTrain1.csv"
    testDataFilename = "./spamTrain2.csv"

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

    # Define percentages of worst features to drop
    drop_percentages = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
    ]

    print(f"\nTesting drop percentages: {drop_percentages}")

    # Analyze feature dropping
    auc_scores, features_kept, features_dropped, sorted_indices, importance = (
        analyze_feature_dropping(
            trainFeatures, trainLabels, testFeatures, testLabels, drop_percentages
        )
    )

    # Create and show plots
    print("\nCreating plots...")

    # Plot 1: AUC and dropped features vs drop percentage
    fig1 = plot_feature_drop_analysis(drop_percentages, auc_scores, features_dropped)
    plt.savefig("trees_drop_analysis.png", dpi=300, bbox_inches="tight")
    print("Drop analysis plot saved as 'trees_drop_analysis.png'")

    # Plot 2: Feature importance
    fig2 = plot_feature_importance(importance, sorted_indices, top_n=20)
    plt.savefig("feature_importance_analysis.png", dpi=300, bbox_inches="tight")
    print("Feature importance plot saved as 'feature_importance_analysis.png'")

    # Show plots
    plt.show()

    # Print summary
    if auc_scores:
        best_auc_idx = np.argmax(auc_scores)
        print(f"\nSummary:")
        print(
            f"Best AUC: {max(auc_scores):.4f} when dropping {drop_percentages[best_auc_idx]}% of worst features"
        )
        print(f"Features kept at best AUC: {features_kept[best_auc_idx]}")
        print(f"Baseline AUC (0% dropped): {auc_scores[0]:.4f}")

        # Find optimal drop percentage (best trade-off)
        auc_improvement = np.array(auc_scores) - auc_scores[0]
        features_reduction = np.array(features_dropped)

        # Simple scoring: AUC improvement per feature dropped
        if len(features_reduction) > 1:
            efficiency_scores = []
            for i in range(1, len(auc_improvement)):
                if features_reduction[i] > 0:
                    efficiency = auc_improvement[i] / features_reduction[i]
                    efficiency_scores.append(efficiency)
                else:
                    efficiency_scores.append(0)

            if efficiency_scores:
                best_efficiency_idx = (
                    np.argmax(efficiency_scores) + 1
                )  # +1 because we started from index 1
                print(
                    f"Most efficient dropping: {drop_percentages[best_efficiency_idx]}% (AUC improvement per feature dropped)"
                )

        # Create detailed results table
        print(f"\nDetailed Results:")
        print(f"{'Drop%':<6} {'AUC':<8} {'Kept':<6} {'Dropped':<8} {'AUC_Change':<10}")
        print("-" * 48)
        for i in range(len(auc_scores)):
            auc_change = auc_scores[i] - auc_scores[0] if i > 0 else 0.0
            print(
                f"{drop_percentages[i]:<6} {auc_scores[i]:<8.4f} {features_kept[i]:<6} {features_dropped[i]:<8} {auc_change:<+10.4f}"
            )

    # Print feature importance summary
    print(f"\nFeature Importance Summary:")
    print(
        f"Most important feature: {sorted_indices[-1]} (importance: {importance[sorted_indices[-1]]:.4f})"
    )
    print(
        f"Least important feature: {sorted_indices[0]} (importance: {importance[sorted_indices[0]]:.4f})"
    )
    print(f"Importance range: {importance.min():.6f} to {importance.max():.4f}")


if __name__ == "__main__":
    main()
