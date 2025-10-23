#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Comparison Graphs
- AUC and TPR comparison across all models
- Validation curves showing training vs validation accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import os
import sys

# Add parent directory to path to import classifySpam
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifySpam import predictTest, predictTestWithTFIDF

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

def get_optimized_params(model_type):
    """Get optimized hyperparameters for each model type"""
    if model_type == "HGB":
        return {
            "learning_rate": 0.1,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_iter": 300,
            "l2_regularization": 0.5,
            "early_stopping": True,
            "validation_fraction": 0.1,
        }
    elif model_type == "RF":
        return {
            "n_estimators": 500,
            "max_features": "sqrt",
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "max_samples": 0.9,
            "n_jobs": -1,
            "random_state": 42,
        }
    elif model_type == "ExtraTrees":
        return {
            "n_estimators": 500,
            "max_features": "sqrt",
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "bootstrap": True,
            "max_samples": 0.9,
            "n_jobs": -1,
            "random_state": 42,
        }
    elif model_type == "KNN":
        return {
            "n_neighbors": 15,
            "weights": "distance",
            "metric": "euclidean",
            "n_jobs": -1,
        }
    elif model_type == "LogReg":
        return {
            "penalty": "l2",
            "C": 10.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    elif model_type == "LogRegTFIDF":
        return {
            "penalty": "l1",
            "C": 50.0,
            "solver": "liblinear",
            "max_iter": 10000,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    return {}

def create_model_pipeline(model_type):
    """Create a pipeline for each model type with optimized parameters"""
    steps = []
    
    # Imputation
    steps.append(("impute", SimpleImputer(missing_values=-1, strategy="median")))
    
    # Scaling (for KNN and LogReg)
    if model_type in ["KNN", "LogReg"]:
        steps.append(("scale", StandardScaler()))
    
    # TF-IDF transformation (for LogRegTFIDF)
    if model_type == "LogRegTFIDF":
        # TF-IDF expects non-negative values, so we need to ensure all values are >= 0
        from sklearn.preprocessing import FunctionTransformer
        def ensure_non_negative(X):
            return np.maximum(X, 0)
        steps.append(("non_neg", FunctionTransformer(ensure_non_negative)))
        steps.append(("tfidf", TfidfTransformer()))
    
    # Model with optimized parameters
    params = get_optimized_params(model_type)
    
    if model_type == "HGB":
        model = HistGradientBoostingClassifier(random_state=42, **params)
    elif model_type == "RF":
        model = RandomForestClassifier(**params)
    elif model_type == "ExtraTrees":
        model = ExtraTreesClassifier(**params)
    elif model_type == "KNN":
        model = KNeighborsClassifier(**params)
    elif model_type == "LogReg":
        model = LogisticRegression(**params)
    elif model_type == "LogRegTFIDF":
        model = LogisticRegression(**params)
    
    steps.append(("model", model))
    
    from sklearn.pipeline import Pipeline
    return Pipeline(steps)

def optimize_and_evaluate_model(model_type, train_features, train_labels, test_features, test_labels, optimize=True):
    """Optimize hyperparameters and evaluate model performance"""
    
    if not optimize:
        # Use pre-defined optimized parameters
        pipeline = create_model_pipeline(model_type)
        
        # Cross-validation on training data only
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, train_features, train_labels, cv=cv, scoring="roc_auc", n_jobs=-1)
        
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
    
    # Perform hyperparameter optimization
    base_pipeline = create_model_pipeline(model_type)
    
    # Define parameter grids for optimization
    param_grids = {
        "HGB": {
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [6, 8, 10],
            "model__min_samples_leaf": [10, 20, 30],
            "model__l2_regularization": [0.1, 0.5, 1.0],
        },
        "RF": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [15, 20, 25],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "KNN": {
            "model__n_neighbors": [5, 10, 15, 20],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
        "LogReg": {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l2"],  # Only L2 for lbfgs solver
            "model__solver": ["lbfgs"],
        },
        "LogRegTFIDF": {
            "model__C": [1.0, 10.0, 50.0, 100.0],
            "model__penalty": ["l1"],
            "model__solver": ["liblinear"],  # Only liblinear for L1 penalty
        },
    }
    
    if model_type not in param_grids:
        # Fallback to non-optimized version
        return optimize_and_evaluate_model(model_type, train_features, train_labels, test_features, test_labels, optimize=False)
    
    # Use RandomizedSearchCV for faster optimization
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Reduced folds for speed
    search = RandomizedSearchCV(
        base_pipeline,
        param_grids[model_type],
        n_iter=20,  # Number of parameter settings sampled
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    print(f"  Optimizing {model_type} hyperparameters...")
    search.fit(train_features, train_labels)
    
    # Get best model and evaluate
    best_pipeline = search.best_estimator_
    cv_scores = search.cv_results_['mean_test_score']
    
    # Test set evaluation
    test_outputs = best_pipeline.predict_proba(test_features)[:, 1]
    test_auc = roc_auc_score(test_labels, test_outputs)
    tpr_at_fpr, fpr, tpr = tprAtFPR(test_labels, test_outputs, 0.01)
    
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV score: {search.best_score_:.4f}")
    
    return {
        "model": model_type,
        "cv_mean": search.best_score_,
        "cv_std": np.std(cv_scores),
        "test_auc": test_auc,
        "tpr_mean": tpr_at_fpr,
        "fpr": fpr,
        "tpr": tpr,
        "best_params": search.best_params_,
    }

def evaluate_model_performance(model_type, train_features, train_labels, test_features, test_labels, optimize=True):
    """Evaluate model performance and return metrics using same approach as evaluateClassifier.py"""
    
    if model_type == "BestModel":
        # Use the actual model from classifySpam.py
        test_outputs = predictTest(train_features, train_labels, test_features)
        test_auc = roc_auc_score(test_labels, test_outputs)
        tpr_at_fpr, fpr, tpr = tprAtFPR(test_labels, test_outputs, 0.01)
        
        # Manual cross-validation for BestModel
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Reduced folds for speed
        cv_scores = []
        
        for train_idx, val_idx in cv.split(train_features, train_labels):
            X_train_cv, X_val_cv = train_features[train_idx], train_features[val_idx]
            y_train_cv, y_val_cv = train_labels[train_idx], train_labels[val_idx]
            
            # Get predictions using predictTest
            val_outputs = predictTest(X_train_cv, y_train_cv, X_val_cv)
            val_auc = roc_auc_score(y_val_cv, val_outputs)
            cv_scores.append(val_auc)
        
        cv_scores = np.array(cv_scores)
        
        return {
            "model": model_type,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "test_auc": test_auc,
            "tpr_mean": tpr_at_fpr,
            "fpr": fpr,
            "tpr": tpr,
        }
    elif model_type == "BestModelTFIDF":
        # Use the TF-IDF enhanced model from classifySpam.py
        test_outputs = predictTestWithTFIDF(train_features, train_labels, test_features)
        test_auc = roc_auc_score(test_labels, test_outputs)
        tpr_at_fpr, fpr, tpr = tprAtFPR(test_labels, test_outputs, 0.01)
        
        # Manual cross-validation for BestModelTFIDF
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Reduced folds for speed
        cv_scores = []
        
        for train_idx, val_idx in cv.split(train_features, train_labels):
            X_train_cv, X_val_cv = train_features[train_idx], train_features[val_idx]
            y_train_cv, y_val_cv = train_labels[train_idx], train_labels[val_idx]
            
            # Get predictions using predictTestWithTFIDF
            val_outputs = predictTestWithTFIDF(X_train_cv, y_train_cv, X_val_cv)
            val_auc = roc_auc_score(y_val_cv, val_outputs)
            cv_scores.append(val_auc)
        
        cv_scores = np.array(cv_scores)
        
        return {
            "model": model_type,
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "test_auc": test_auc,
            "tpr_mean": tpr_at_fpr,
            "fpr": fpr,
            "tpr": tpr,
        }
    else:
        # Use the optimization function for other models
        return optimize_and_evaluate_model(model_type, train_features, train_labels, test_features, test_labels, optimize)

def create_model_performance_graph():
    """Create graph showing model performance in terms of AUC and TPR"""
    print("=" * 80)
    print("MODEL PERFORMANCE COMPARISON (AUC & TPR)")
    print("=" * 80)
    
    # Load data using same approach as evaluateClassifier.py
    train_data_filename = "../spamTrain1.csv"
    test_data_filename = "../spamTrain2.csv"
    
    train_data = np.loadtxt(train_data_filename, delimiter=",")
    test_data = np.loadtxt(test_data_filename, delimiter=",")
    
    # Separate labels (last column) from training and test data
    train_features = train_data[:, :-1]
    train_labels = train_data[:, -1]
    test_features = test_data[:, :-1]
    test_labels = test_data[:, -1]
    
    print(f"Train: {train_features.shape[0]} samples, {train_features.shape[1]} features")
    print(f"Test: {test_features.shape[0]} samples")
    
    # Evaluate all models (including the best model from classifySpam.py and TF-IDF enhanced version)
    models = ["BestModel", "BestModelTFIDF", "HGB", "RF", "KNN", "LogReg", "LogRegTFIDF"]
    results = []
    
    for model_type in models:
        print(f"\nEvaluating {model_type}...")
        if model_type not in ["BestModel", "BestModelTFIDF"]:
            print(f"  Using hyperparameter optimization for {model_type}")
        result = evaluate_model_performance(
            model_type, train_features, train_labels, test_features, test_labels, optimize=True
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
    
    # Load data using same approach as evaluateClassifier.py
    train_data_filename = "../spamTrain1.csv"
    train_data = np.loadtxt(train_data_filename, delimiter=",")
    features = train_data[:, :-1]
    labels = train_data[:, -1]
    
    print(f"Training dataset: {features.shape[0]} samples, {features.shape[1]} features")
    
    # Setup learning curve parameters
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)
    models = ["RF", "ExtraTrees"]  # Only RF and Extra Trees for learning curves
    colors = {"RF": "green", "ExtraTrees": "purple"}
    
    # Create subplots for RF and Extra Trees (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
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
        val_color = "darkred" if model_type == "RF" else "darkblue"
        ax.plot(sizes, val_acc_mean, "o-", color=val_color, linewidth=2.5,
                label="Validation Accuracy", markersize=6)
        ax.fill_between(sizes, val_acc_mean - val_acc_std,
                       val_acc_mean + val_acc_std, alpha=0.2, color=val_color)
        
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
