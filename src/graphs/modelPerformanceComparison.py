#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Comparison Graphs
- AUC and TPR comparison across all models
- Validation curves showing training vs validation accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve
import os

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

def create_model_pipeline(model_type):
    """Create a pipeline for each model type"""
    steps = []
    
    # Imputation
    steps.append(("impute", SimpleImputer(missing_values=-1, strategy="median")))
    
    # Scaling (for KNN and LogReg)
    if model_type in ["KNN", "LogReg"]:
        steps.append(("scale", StandardScaler()))
    
    # Model
    if model_type == "HGB":
        model = HistGradientBoostingClassifier(
            random_state=42,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=50,
            max_iter=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
        )
    elif model_type == "RF":
        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            max_samples=0.8,
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=21,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )
    elif model_type == "LogReg":
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    
    steps.append(("model", model))
    
    from sklearn.pipeline import Pipeline
    return Pipeline(steps)

def evaluate_model_performance(model_type, features, labels, train_features, train_labels, test_features, test_labels):
    """Evaluate model performance and return metrics"""
    pipeline = create_model_pipeline(model_type)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, features, labels, cv=cv, scoring="roc_auc", n_jobs=-1)
    
    # Test set evaluation
    pipeline.fit(train_features, train_labels)
    test_outputs = pipeline.predict_proba(test_features)[:, 1]
    test_auc = roc_auc_score(test_labels, test_outputs)
    tpr_at_fpr, fpr, tpr = tprAtFPR(test_labels, test_outputs, 0.01)
    
    return {
        "model": model_type,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "test_auc": test_auc,
        "tpr_mean": tpr_at_fpr,
        "fpr": fpr,
        "tpr": tpr,
    }

def create_model_performance_graph():
    """Create graph showing model performance in terms of AUC and TPR"""
    print("=" * 80)
    print("MODEL PERFORMANCE COMPARISON (AUC & TPR)")
    print("=" * 80)
    
    # Load data
    data_filename = "../spamTrain1.csv"
    data = np.loadtxt(data_filename, delimiter=",")
    features = data[:, :-1]
    labels = data[:, -1]
    
    train_features = features[0::2, :]
    train_labels = labels[0::2]
    test_features = features[1::2, :]
    test_labels = labels[1::2]
    
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Train: {train_features.shape[0]} | Test: {test_features.shape[0]}")
    
    # Evaluate all models
    models = ["HGB", "RF", "KNN", "LogReg"]
    results = []
    
    for model_type in models:
        print(f"\nEvaluating {model_type}...")
        result = evaluate_model_performance(
            model_type, features, labels, train_features, train_labels, test_features, test_labels
        )
        results.append(result)
        print(f"  CV AUC: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
        print(f"  Test AUC: {result['test_auc']:.4f}")
        print(f"  TPR@0.01: {result['tpr_mean']:.4f}")
    
    # Create performance comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: AUC Comparison
    ax1 = axes[0]
    model_names = [r["model"] for r in results]
    cv_aucs = [r["cv_mean"] for r in results]
    cv_stds = [r["cv_std"] for r in results]
    test_aucs = [r["test_auc"] for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cv_aucs, width, yerr=cv_stds, 
                   label='CV AUC', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x + width/2, test_aucs, width, 
                   label='Test AUC', alpha=0.8)
    
    ax1.set_ylabel('AUC Score', fontsize=12)
    ax1.set_title('Model Performance: AUC Scores', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim([0.4, 1.0])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: TPR@0.01 Comparison
    ax2 = axes[1]
    tprs = [r["tpr_mean"] for r in results]
    
    bars3 = ax2.bar(x, tprs, alpha=0.8, color=['blue', 'green', 'red', 'purple'])
    
    ax2.set_ylabel('TPR @ FPR=0.01', fontsize=12)
    ax2.set_title('Model Performance: TPR at Low FPR', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=11)
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    output_path = "../../images/model_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Performance comparison saved: {output_path}")
    plt.show()
    
    return results

def create_learning_curves_graph():
    """Create validation curves showing training vs validation accuracy vs training examples"""
    print("\n" + "=" * 80)
    print("LEARNING CURVES (Training vs Validation Accuracy)")
    print("=" * 80)
    
    # Load data
    data_filename = "../spamTrain1.csv"
    data = np.loadtxt(data_filename, delimiter=",")
    features = data[:, :-1]
    labels = data[:, -1]
    
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Setup learning curve parameters
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)
    models = ["HGB", "RF", "KNN", "LogReg"]
    colors = {"HGB": "blue", "RF": "green", "KNN": "red", "LogReg": "purple"}
    
    # Create subplots for all models
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, model_type in enumerate(models):
        print(f"\nGenerating learning curve for {model_type}...")
        
        # Create pipeline
        pipeline = create_model_pipeline(model_type)
        
        # Generate learning curves for Accuracy
        sizes, train_scores_acc, val_scores_acc = learning_curve(
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
        train_acc_mean = np.mean(train_scores_acc, axis=1)
        train_acc_std = np.std(train_scores_acc, axis=1)
        val_acc_mean = np.mean(val_scores_acc, axis=1)
        val_acc_std = np.std(val_scores_acc, axis=1)
        
        # Plot learning curves
        ax = axes[idx]
        color = colors[model_type]
        
        # Training accuracy
        ax.plot(sizes, train_acc_mean, "o-", color=color, linewidth=2.5,
                label="Training Accuracy", markersize=6)
        ax.fill_between(sizes, train_acc_mean - train_acc_std,
                       train_acc_mean + train_acc_std, alpha=0.2, color=color)
        
        # Validation accuracy
        ax.plot(sizes, val_acc_mean, "o-", color="orange", linewidth=2.5,
                label="Validation Accuracy", markersize=6)
        ax.fill_between(sizes, val_acc_mean - val_acc_std,
                       val_acc_mean + val_acc_std, alpha=0.2, color="orange")
        
        ax.set_xlabel("Number of Training Examples", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{model_type} Learning Curve", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.05])
        
        # Add final values
        final_train_acc = train_acc_mean[-1]
        final_val_acc = val_acc_mean[-1]
        gap = final_train_acc - final_val_acc
        
        ax.text(0.02, 0.02, 
               f"Final Train: {final_train_acc:.3f}\nFinal Val: {final_val_acc:.3f}\nGap: {gap:.3f}",
               transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        
        print(f"  Final Training Accuracy: {final_train_acc:.4f}")
        print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"  Overfitting Gap: {gap:.4f}")
    
    plt.suptitle("Learning Curves: Training vs Validation Accuracy", 
                fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Save plot
    output_path = "../../images/learning_curves_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Learning curves saved: {output_path}")
    plt.show()

def main():
    """Main function to create both graphs"""
    print("Creating Model Performance Comparison Graphs...")
    
    # Change to graphs directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create model performance comparison
    results = create_model_performance_graph()
    
    # Create learning curves
    create_learning_curves_graph()
    
    print("\n" + "=" * 80)
    print("✅ ALL GRAPHS CREATED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("- ../../images/model_performance_comparison.png - AUC and TPR comparison")
    print("- ../../images/learning_curves_comparison.png - Learning curves comparison")
    
    # Print summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 50)
    print(f"{'Model':<10} {'Test AUC':<12} {'TPR@0.01':<12}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["test_auc"], reverse=True):
        print(f"{r['model']:<10} {r['test_auc']:<12.4f} {r['tpr_mean']:<12.4f}")

if __name__ == "__main__":
    main()
