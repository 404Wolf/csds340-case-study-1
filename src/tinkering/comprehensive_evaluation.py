# -*- coding: utf-8 -*-
"""
Comprehensive Spam Classification Evaluation Suite
- Interactive configuration selection
- Model parameter customization
- Learning curves (train vs validation)
- Full model comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline
import os
from datetime import datetime


# ============================================================================
# CONFIGURATION AND PIPELINE CREATION
# ============================================================================


def create_pipeline(
    model_type, use_fs, impute_strategy, scale_method, model_params=None
):
    """Create a pipeline with specified configuration"""
    steps = []

    # 1. Imputation
    if impute_strategy == "mean":
        steps.append(("impute", SimpleImputer(missing_values=-1, strategy="mean")))
    elif impute_strategy == "median":
        steps.append(("impute", SimpleImputer(missing_values=-1, strategy="median")))
    elif impute_strategy == "knn":
        steps.append(("impute", KNNImputer(missing_values=-1, n_neighbors=5)))
    elif impute_strategy == "iterative":
        steps.append(("impute", IterativeImputer(missing_values=-1, random_state=42)))

    # 2. Remove constant features
    steps.append(("nzvar", VarianceThreshold(threshold=0.0)))

    # 3. Scaling
    if scale_method == "standard":
        steps.append(("scale", StandardScaler()))
    elif scale_method == "robust":
        steps.append(("scale", RobustScaler()))
    elif scale_method == "minmax":
        steps.append(("scale", MinMaxScaler()))

    # 4. Feature Selection
    if use_fs:
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

    # 5. Model with default or custom parameters
    if model_params is None:
        model_params = {}

    if model_type == "HGB":
        model = HistGradientBoostingClassifier(
            random_state=42,
            learning_rate=model_params.get("learning_rate", 0.05),
            max_depth=model_params.get("max_depth", 5),
            min_samples_leaf=model_params.get("min_samples_leaf", 50),
            max_iter=model_params.get("max_iter", 200),
            l2_regularization=model_params.get("l2_regularization", 1.0),
            early_stopping=model_params.get("early_stopping", True),
            validation_fraction=model_params.get("validation_fraction", 0.1),
        )
    elif model_type == "RF":
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 300),
            max_features=model_params.get("max_features", "sqrt"),
            max_depth=model_params.get("max_depth", 15),
            min_samples_split=model_params.get("min_samples_split", 10),
            min_samples_leaf=model_params.get("min_samples_leaf", 5),
            class_weight=model_params.get("class_weight", "balanced"),
            max_samples=model_params.get("max_samples", 0.8),
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=model_params.get("n_neighbors", 21),
            weights=model_params.get("weights", "distance"),
            metric=model_params.get("metric", "euclidean"),
            n_jobs=-1,
        )
    elif model_type == "LogReg":
        model = LogisticRegression(
            penalty=model_params.get("penalty", "l2"),
            C=model_params.get("C", 1.0),
            solver=model_params.get("solver", "lbfgs"),
            max_iter=model_params.get("max_iter", 1000),
            class_weight=model_params.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
        )

    steps.append(("model", model))
    return Pipeline(steps)


def get_model_parameters_interactive(model_type):
    """Interactively get model parameters"""
    print(f"\n--- {model_type} Parameters ---")
    print("Press Enter to use default values")

    params = {}

    if model_type == "HGB":
        lr = input(f"  Learning rate [default: 0.05]: ").strip()
        if lr:
            params["learning_rate"] = float(lr)

        depth = input(f"  Max depth [default: 5]: ").strip()
        if depth:
            params["max_depth"] = int(depth)

        leaf = input(f"  Min samples leaf [default: 50]: ").strip()
        if leaf:
            params["min_samples_leaf"] = int(leaf)

        l2 = input(f"  L2 regularization [default: 1.0]: ").strip()
        if l2:
            params["l2_regularization"] = float(l2)

    elif model_type == "RF":
        trees = input(f"  Number of trees [default: 300]: ").strip()
        if trees:
            params["n_estimators"] = int(trees)

        depth = input(f"  Max depth [default: 15, None for unlimited]: ").strip()
        if depth:
            params["max_depth"] = None if depth.lower() == "none" else int(depth)

        split = input(f"  Min samples split [default: 10]: ").strip()
        if split:
            params["min_samples_split"] = int(split)

        leaf = input(f"  Min samples leaf [default: 5]: ").strip()
        if leaf:
            params["min_samples_leaf"] = int(leaf)

    elif model_type == "KNN":
        k = input(f"  Number of neighbors (k) [default: 21]: ").strip()
        if k:
            params["n_neighbors"] = int(k)

        weights = input(f"  Weights (uniform/distance) [default: distance]: ").strip()
        if weights:
            params["weights"] = weights

    elif model_type == "LogReg":
        c = input(f"  C (inverse regularization) [default: 1.0]: ").strip()
        if c:
            params["C"] = float(c)

        penalty = input(f"  Penalty (l1/l2) [default: l2]: ").strip()
        if penalty:
            params["penalty"] = penalty

    return params


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def tprAtFPR(labels, outputs, desiredFPR):
    """Calculate TPR at specific FPR"""
    fpr, tpr, _ = roc_curve(labels, outputs)
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow, fprAbove = fpr[maxFprIndex], fpr[maxFprIndex + 1]
    tprBelow, tprAbove = tpr[maxFprIndex], tpr[maxFprIndex + 1]
    tprAt = (tprAbove - tprBelow) / (fprAbove - fprBelow) * (
        desiredFPR - fprBelow
    ) + tprBelow
    return tprAt, fpr, tpr


def evaluate_model(
    model_type,
    features,
    labels,
    trainFeatures,
    trainLabels,
    testFeatures,
    testLabels,
    use_fs,
    impute_strategy,
    scale_method,
    model_params,
    n_runs=3,
):
    """Evaluate a single model with multiple runs"""
    cv_scores_all = []
    test_aucs = []
    test_tprs = []
    fprs_list = []
    tprs_list = []

    for run in range(n_runs):
        # Create pipeline
        pipeline = create_pipeline(
            model_type, use_fs, impute_strategy, scale_method, model_params
        )

        # Update random state for this run
        if hasattr(pipeline.named_steps["model"], "random_state"):
            pipeline.named_steps["model"].random_state = 42 + run

        # Cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + run)
        cv_scores = cross_val_score(
            pipeline, features, labels, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        cv_scores_all.extend(cv_scores)

        # Test set evaluation
        pipeline.fit(trainFeatures, trainLabels)
        testOutputs = pipeline.predict_proba(testFeatures)[:, 1]
        test_auc = roc_auc_score(testLabels, testOutputs)
        tpr_at_fpr, fpr, tpr = tprAtFPR(testLabels, testOutputs, 0.01)

        test_aucs.append(test_auc)
        test_tprs.append(tpr_at_fpr)
        fprs_list.append(fpr)
        tprs_list.append(tpr)

    return {
        "model": model_type,
        "cv_mean": np.mean(cv_scores_all),
        "cv_std": np.std(cv_scores_all),
        "test_auc_mean": np.mean(test_aucs),
        "test_auc_std": np.std(test_aucs),
        "tpr_mean": np.mean(test_tprs),
        "tpr_std": np.std(test_tprs),
        "fpr": fprs_list[0],
        "tpr": tprs_list[0],
    }


# ============================================================================
# LEARNING CURVES
# ============================================================================


def create_output_folder(use_fs, impute_strategy, scale_method):
    """Create output folder based on configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fs_str = "FS" if use_fs else "NoFS"
    config_str = f"{fs_str}_{impute_strategy}_{scale_method}"
    folder_name = f"results_{config_str}_{timestamp}"

    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def generate_learning_curves(
    models_config,
    features,
    labels,
    use_fs,
    impute_strategy,
    scale_method,
    output_folder,
):
    """Generate learning curves for all models"""
    print("\n" + "=" * 80)
    print("GENERATING LEARNING CURVES (Training vs Validation)")
    print("=" * 80)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)
    colors = {"HGB": "blue", "RF": "green", "KNN": "red", "LogReg": "purple"}

    all_gaps = []

    for model_type, model_params in models_config.items():
        print(f"\n{model_type}...")

        # Create pipeline
        pipeline = create_pipeline(
            model_type, use_fs, impute_strategy, scale_method, model_params
        )

        # Generate learning curves for AUC
        sizes, train_scores_auc, val_scores_auc = learning_curve(
            pipeline,
            features,
            labels,
            cv=cv,
            n_jobs=-1,
            train_sizes=train_sizes,
            scoring="roc_auc",
            shuffle=True,
            random_state=42,
        )

        # Generate learning curves for Accuracy
        _, train_scores_acc, val_scores_acc = learning_curve(
            pipeline,
            features,
            labels,
            cv=cv,
            n_jobs=-1,
            train_sizes=train_sizes,
            scoring="accuracy",
            shuffle=True,
            random_state=42,
        )

        # Calculate means and stds
        train_auc_mean = np.mean(train_scores_auc, axis=1)
        train_auc_std = np.std(train_scores_auc, axis=1)
        val_auc_mean = np.mean(val_scores_auc, axis=1)
        val_auc_std = np.std(val_scores_auc, axis=1)

        train_acc_mean = np.mean(train_scores_acc, axis=1)
        train_acc_std = np.std(train_scores_acc, axis=1)
        val_acc_mean = np.mean(val_scores_acc, axis=1)
        val_acc_std = np.std(val_scores_acc, axis=1)

        # Plot individual learning curve (2x2 grid)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Top Left: AUC Learning Curves
        ax1.plot(
            sizes,
            train_auc_mean,
            "o-",
            color="red",
            linewidth=2.5,
            label="Training AUC",
            markersize=8,
        )
        ax1.fill_between(
            sizes,
            train_auc_mean - train_auc_std,
            train_auc_mean + train_auc_std,
            alpha=0.2,
            color="red",
        )

        ax1.plot(
            sizes,
            val_auc_mean,
            "o-",
            color="green",
            linewidth=2.5,
            label="Validation AUC",
            markersize=8,
        )
        ax1.fill_between(
            sizes,
            val_auc_mean - val_auc_std,
            val_auc_mean + val_auc_std,
            alpha=0.2,
            color="green",
        )

        ax1.set_xlabel("Training Examples", fontsize=11)
        ax1.set_ylabel("AUC Score", fontsize=11)
        ax1.set_title(
            f"AUC Learning Curve - {model_type}", fontsize=12, fontweight="bold"
        )
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.4, 1.05])

        final_train_auc = train_auc_mean[-1]
        final_val_auc = val_auc_mean[-1]
        ax1.text(
            0.02,
            0.02,
            f"Final Train: {final_train_auc:.3f}\nFinal Val: {final_val_auc:.3f}",
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Top Right: Accuracy Learning Curves
        ax2.plot(
            sizes,
            train_acc_mean,
            "o-",
            color="darkred",
            linewidth=2.5,
            label="Training Accuracy",
            markersize=8,
        )
        ax2.fill_between(
            sizes,
            train_acc_mean - train_acc_std,
            train_acc_mean + train_acc_std,
            alpha=0.2,
            color="darkred",
        )

        ax2.plot(
            sizes,
            val_acc_mean,
            "o-",
            color="darkgreen",
            linewidth=2.5,
            label="Validation Accuracy",
            markersize=8,
        )
        ax2.fill_between(
            sizes,
            val_acc_mean - val_acc_std,
            val_acc_mean + val_acc_std,
            alpha=0.2,
            color="darkgreen",
        )

        ax2.set_xlabel("Training Examples", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.set_title(
            f"Accuracy Learning Curve - {model_type}", fontsize=12, fontweight="bold"
        )
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.4, 1.05])

        final_train_acc = train_acc_mean[-1]
        final_val_acc = val_acc_mean[-1]
        ax2.text(
            0.02,
            0.02,
            f"Final Train: {final_train_acc:.3f}\nFinal Val: {final_val_acc:.3f}",
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        # Bottom Left: AUC Overfitting Gap
        gap_auc = train_auc_mean - val_auc_mean
        final_gap_auc = gap_auc[-1]

        ax3.plot(
            sizes,
            gap_auc,
            "o-",
            color="orange",
            linewidth=2.5,
            label="AUC Gap",
            markersize=8,
        )
        ax3.fill_between(sizes, 0, gap_auc, alpha=0.3, color="orange")
        ax3.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax3.axhline(
            y=0.05,
            color="green",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Good fit",
        )
        ax3.axhline(
            y=0.15,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Overfitting",
        )

        ax3.set_xlabel("Training Examples", fontsize=11)
        ax3.set_ylabel("AUC Gap (Train - Val)", fontsize=11)
        ax3.set_title("AUC Overfitting Analysis", fontsize=12, fontweight="bold")
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(True, alpha=0.3)

        if final_gap_auc < 0.05:
            status_auc = "✓ Good fit"
            status_color_auc = "green"
        elif final_gap_auc < 0.15:
            status_auc = "⚠ Mild overfitting"
            status_color_auc = "orange"
        else:
            status_auc = "✗ Significant overfitting"
            status_color_auc = "red"

        ax3.text(
            0.98,
            0.98,
            f"Gap: {final_gap_auc:.3f}\n{status_auc}",
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor=status_color_auc, alpha=0.3),
        )

        # Bottom Right: Accuracy Overfitting Gap
        gap_acc = train_acc_mean - val_acc_mean
        final_gap_acc = gap_acc[-1]

        ax4.plot(
            sizes,
            gap_acc,
            "o-",
            color="purple",
            linewidth=2.5,
            label="Accuracy Gap",
            markersize=8,
        )
        ax4.fill_between(sizes, 0, gap_acc, alpha=0.3, color="purple")
        ax4.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax4.axhline(
            y=0.05,
            color="green",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Good fit",
        )
        ax4.axhline(
            y=0.15,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Overfitting",
        )

        ax4.set_xlabel("Training Examples", fontsize=11)
        ax4.set_ylabel("Accuracy Gap (Train - Val)", fontsize=11)
        ax4.set_title("Accuracy Overfitting Analysis", fontsize=12, fontweight="bold")
        ax4.legend(loc="best", fontsize=9)
        ax4.grid(True, alpha=0.3)

        if final_gap_acc < 0.05:
            status_acc = "✓ Good fit"
            status_color_acc = "green"
        elif final_gap_acc < 0.15:
            status_acc = "⚠ Mild overfitting"
            status_color_acc = "orange"
        else:
            status_acc = "✗ Significant overfitting"
            status_color_acc = "red"

        ax4.text(
            0.98,
            0.98,
            f"Gap: {final_gap_acc:.3f}\n{status_acc}",
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor=status_color_acc, alpha=0.3),
        )

        final_gap = final_gap_auc  # Use AUC gap for summary
        status = status_auc

        plt.suptitle(
            f"{model_type} - Training vs Validation Performance",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        filename = os.path.join(
            output_folder, f"learning_curve_{model_type.lower()}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {filename}")
        print(
            f"  AUC gap: {final_gap_auc:.4f} ({status_auc}) | Accuracy gap: {final_gap_acc:.4f} ({status_acc})"
        )
        plt.close()

        all_gaps.append(
            {"model": model_type, "gap": final_gap, "color": colors[model_type]}
        )

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_gaps))
    gaps = [g["gap"] for g in all_gaps]
    gap_colors = [g["color"] for g in all_gaps]
    names = [g["model"] for g in all_gaps]

    bars = ax.bar(x, gaps, color=gap_colors, alpha=0.7, edgecolor="black", linewidth=2)

    ax.axhline(
        y=0.05, color="green", linestyle="--", linewidth=2, alpha=0.7, label="Good fit"
    )
    ax.axhline(
        y=0.15,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Mild overfitting",
    )

    ax.set_ylabel("Train-Validation AUC Gap", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Overfitting Comparison Across All Models", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{gap:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    filename = os.path.join(output_folder, "overfitting_summary.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n📊 Summary saved: {filename}")
    plt.close()


# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================


def plot_comparison(results, config_name, output_folder):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"HGB": "blue", "RF": "green", "KNN": "red", "LogReg": "purple"}

    # ROC Curves (Full)
    ax1 = axes[0]
    for r in results:
        ax1.plot(
            r["fpr"],
            r["tpr"],
            color=colors[r["model"]],
            linewidth=2.5,
            label=f"{r['model']} (AUC={r['test_auc_mean']:.3f})",
        )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.3)
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curves", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ROC Curves (Zoomed)
    ax2 = axes[1]
    for r in results:
        mask = r["fpr"] <= 0.1
        ax2.plot(
            r["fpr"][mask],
            r["tpr"][mask],
            color=colors[r["model"]],
            linewidth=2.5,
            label=f"{r['model']} (TPR@0.01={r['tpr_mean']:.3f})",
        )
    ax2.axvline(x=0.01, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("ROC Curves (Low FPR Region)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.1])

    # Metrics Comparison
    ax3 = axes[2]
    x = np.arange(len(results))
    width = 0.35

    test_aucs = [r["test_auc_mean"] for r in results]
    tprs = [r["tpr_mean"] for r in results]
    model_colors = [colors[r["model"]] for r in results]
    names = [r["model"] for r in results]

    bars1 = ax3.bar(
        x - width / 2,
        test_aucs,
        width,
        label="Test AUC",
        color=model_colors,
        alpha=0.7,
        edgecolor="black",
    )
    bars2 = ax3.bar(
        x + width / 2,
        tprs,
        width,
        label="TPR @ FPR=0.01",
        color=model_colors,
        alpha=0.4,
        edgecolor="black",
    )

    ax3.set_ylabel("Score", fontsize=11)
    ax3.set_title("Performance Metrics", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle(f"Results: {config_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    filename = os.path.join(output_folder, "model_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"📊 Comparison saved: {filename}")
    plt.close()


# ============================================================================
# MAIN PROGRAM
# ============================================================================


def main():
    # Load data
    dataFilename = "../src/spamTrain1.csv"
    data = np.loadtxt(dataFilename, delimiter=",")
    features = data[:, :-1]
    labels = data[:, -1]

    trainFeatures = features[0::2, :]
    trainLabels = labels[0::2]
    testFeatures = features[1::2, :]
    testLabels = labels[1::2]

    print("=" * 80)
    print("COMPREHENSIVE SPAM CLASSIFICATION EVALUATION SUITE")
    print("=" * 80)
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Train: {trainFeatures.shape[0]} | Test: {testFeatures.shape[0]}\n")

    # ========== PREPROCESSING CONFIGURATION ==========
    print("STEP 1: PREPROCESSING CONFIGURATION")
    print("-" * 80)

    # Feature Selection
    print("\n1. Feature Selection:")
    print("   [1] Without feature selection")
    print("   [2] With RF-based feature selection (top 50%)")
    fs_choice = input("Choose (1 or 2): ").strip()
    use_fs = fs_choice == "2"

    # Imputation
    print("\n2. Imputation Strategy:")
    print("   [1] Mean")
    print("   [2] Median (recommended for trees)")
    print("   [3] KNN")
    print("   [4] Iterative (MICE)")
    impute_choice = input("Choose (1-4): ").strip()
    impute_map = {"1": "mean", "2": "median", "3": "knn", "4": "iterative"}
    impute_strategy = impute_map.get(impute_choice, "median")

    # Scaling
    print("\n3. Scaling Method:")
    print("   [1] None (for tree-based)")
    print("   [2] Standard (mean=0, std=1)")
    print("   [3] Robust (median/IQR)")
    print("   [4] MinMax ([0,1])")
    scale_choice = input("Choose (1-4): ").strip()
    scale_map = {"1": "none", "2": "standard", "3": "robust", "4": "minmax"}
    scale_method = scale_map.get(scale_choice, "none")

    # Number of runs
    print("\n4. Number of runs per model (for robustness):")
    n_runs_input = input("Enter (default 3): ").strip()
    n_runs = int(n_runs_input) if n_runs_input.isdigit() else 3

    # ========== MODEL PARAMETERS ==========
    print("\n" + "=" * 80)
    print("STEP 2: MODEL PARAMETERS (Press Enter for defaults)")
    print("-" * 80)

    models_to_run = ["HGB", "RF", "KNN", "LogReg"]
    models_config = {}

    customize = input("\nCustomize model parameters? (y/n) [n]: ").strip().lower()

    if customize == "y":
        for model_type in models_to_run:
            models_config[model_type] = get_model_parameters_interactive(model_type)
    else:
        for model_type in models_to_run:
            models_config[model_type] = {}

    # ========== CREATE OUTPUT FOLDER ==========
    output_folder = create_output_folder(use_fs, impute_strategy, scale_method)

    # ========== CONFIGURATION SUMMARY ==========
    config_name = f"FS={use_fs}, Impute={impute_strategy}, Scale={scale_method}"
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Feature Selection: {use_fs}")
    print(f"Imputation: {impute_strategy}")
    print(f"Scaling: {scale_method}")
    print(f"Runs per model: {n_runs}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Output folder: {output_folder}")
    print("=" * 80)

    input("\nPress Enter to start evaluation...")

    # ========== EVALUATION ==========
    print("\n" + "=" * 80)
    print("EVALUATING MODELS...")
    print("=" * 80)

    results = []
    for model_type in models_to_run:
        print(f"\n{model_type}...")
        result = evaluate_model(
            model_type,
            features,
            labels,
            trainFeatures,
            trainLabels,
            testFeatures,
            testLabels,
            use_fs,
            impute_strategy,
            scale_method,
            models_config[model_type],
            n_runs,
        )
        results.append(result)
        print(f"  CV AUC: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(
            f"  Test AUC: {result['test_auc_mean']:.4f} ± {result['test_auc_std']:.4f}"
        )
        print(f"  TPR@0.01: {result['tpr_mean']:.4f} ± {result['tpr_std']:.4f}")

    # ========== RESULTS SUMMARY ==========
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<10} {'CV AUC':<22} {'Test AUC':<22} {'TPR@0.01':<20}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["test_auc_mean"], reverse=True):
        print(
            f"{r['model']:<10} "
            f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}     "
            f"{r['test_auc_mean']:.4f} ± {r['test_auc_std']:.4f}     "
            f"{r['tpr_mean']:.4f} ± {r['tpr_std']:.4f}"
        )

    best = max(results, key=lambda x: x["test_auc_mean"])
    print("\n" + "=" * 80)
    print(f"🏆 BEST: {best['model']} (AUC: {best['test_auc_mean']:.4f})")
    print("=" * 80)

    # ========== VISUALIZATIONS ==========
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)

    # ROC comparison
    print("\nCreating ROC comparison plots...")
    plot_comparison(results, config_name, output_folder)

    # Learning curves
    print("\nGenerating learning curves...")
    generate_learning_curves(
        models_config,
        features,
        labels,
        use_fs,
        impute_strategy,
        scale_method,
        output_folder,
    )

    # Save configuration to text file
    config_file = os.path.join(output_folder, "configuration.txt")
    with open(config_file, "w") as f:
        f.write("CONFIGURATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Feature Selection: {use_fs}\n")
        f.write(f"Imputation: {impute_strategy}\n")
        f.write(f"Scaling: {scale_method}\n")
        f.write(f"Runs per model: {n_runs}\n")
        f.write(f"\nMODEL PARAMETERS\n")
        f.write("=" * 60 + "\n")
        for model_type, params in models_config.items():
            f.write(f"\n{model_type}:\n")
            if params:
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write("  (using defaults)\n")
        f.write(f"\n\nRESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Model':<10} {'CV AUC':<22} {'Test AUC':<22} {'TPR@0.01':<20}\n")
        f.write("-" * 60 + "\n")
        for r in sorted(results, key=lambda x: x["test_auc_mean"], reverse=True):
            f.write(
                f"{r['model']:<10} "
                f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}     "
                f"{r['test_auc_mean']:.4f} ± {r['test_auc_std']:.4f}     "
                f"{r['tpr_mean']:.4f} ± {r['tpr_std']:.4f}\n"
            )
    print(f"📄 Configuration saved: {config_file}")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All results saved to: {output_folder}/")
    print("\nGenerated files:")
    print("- configuration.txt - Configuration and results summary")
    print("- model_comparison.png - ROC curves and metrics comparison")
    print("- learning_curve_*.png - Individual model learning curves")
    print("- overfitting_summary.png - Overfitting comparison")
    print(f"\n💡 Tip: Keep this folder to compare with future runs!")


if __name__ == "__main__":
    main()
