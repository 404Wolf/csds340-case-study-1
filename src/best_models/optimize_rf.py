#!/usr/bin/env python3
"""
Comprehensive Random Forest Optimization Script
Maximizes RF performance through:
- Hyperparameter tuning
- Different RF variants (RandomForest, ExtraTrees)
- Feature selection strategies
- Ensemble methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFECV
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer
from scipy.stats import randint, uniform
import time
import os
from datetime import datetime
import pickle


def tprAtFPR(labels, outputs, desiredFPR=0.01):
    """Calculate TPR at a specific FPR"""
    fpr, tpr, _ = roc_curve(labels, outputs)
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = (tprAbove - tprBelow) / (fprAbove - fprBelow) * (desiredFPR - fprBelow) + tprBelow
    return tprAt, fpr, tpr


def create_output_folder():
    """Create timestamped output folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"best_models/rf_optimization_{timestamp}"
    os.makedirs(folder, exist_ok=True)
    return folder


# Strategy 1: Comprehensive Hyperparameter Search
def strategy_random_forest_exhaustive(features, labels, cv):
    """Exhaustive hyperparameter search for standard Random Forest"""
    print("\n" + "="*70)
    print("STRATEGY 1: Random Forest - Exhaustive Hyperparameter Search")
    print("="*70)
    
    # Define parameter distributions for randomized search
    param_distributions = {
        'model__n_estimators': [300, 500, 700, 1000],
        'model__max_depth': [10, 15, 20, 25, 30, None],
        'model__min_samples_split': [2, 5, 8, 10, 15],
        'model__min_samples_leaf': [1, 2, 4, 5, 8],
        'model__max_features': ['sqrt', 'log2', 0.3, 0.5],
        'model__max_samples': [0.7, 0.8, 0.9, 1.0],
        'model__criterion': ['gini', 'entropy'],
        'model__class_weight': ['balanced', 'balanced_subsample', None],
        'model__bootstrap': [True],
    }
    
    # Base pipeline
    base_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='median'
        )),
        ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Randomized search (faster than grid search)
    print("Running RandomizedSearchCV (100 iterations)...")
    random_search = RandomizedSearchCV(
        base_pipeline,
        param_distributions,
        n_iter=100,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    start = time.time()
    random_search.fit(features, labels)
    elapsed = time.time() - start
    
    print(f"\n✓ Search completed in {elapsed/60:.1f} minutes")
    print(f"Best CV AUC: {random_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'name': 'RF_Exhaustive',
        'pipeline': random_search.best_estimator_,
        'cv_auc': random_search.best_score_,
        'params': random_search.best_params_,
        'search_time': elapsed
    }


# Strategy 2: ExtraTrees (More Randomization)
def strategy_extra_trees(features, labels, cv):
    """Try ExtraTreesClassifier for comparison"""
    print("\n" + "="*70)
    print("STRATEGY 2: ExtraTrees - Extreme Randomization")
    print("="*70)
    
    param_distributions = {
        'model__n_estimators': [500, 700, 1000],
        'model__max_depth': [15, 20, 25, 30, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', 0.3],
        'model__criterion': ['gini', 'entropy'],
        'model__class_weight': ['balanced', None],
    }
    
    base_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='median'
        )),
        ('model', ExtraTreesClassifier(random_state=42, n_jobs=-1, bootstrap=True))
    ])
    
    print("Running RandomizedSearchCV (80 iterations)...")
    random_search = RandomizedSearchCV(
        base_pipeline,
        param_distributions,
        n_iter=80,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    start = time.time()
    random_search.fit(features, labels)
    elapsed = time.time() - start
    
    print(f"\n✓ Search completed in {elapsed/60:.1f} minutes")
    print(f"Best CV AUC: {random_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'name': 'ExtraTrees',
        'pipeline': random_search.best_estimator_,
        'cv_auc': random_search.best_score_,
        'params': random_search.best_params_,
        'search_time': elapsed
    }


# Strategy 3: Feature Selection Variants
def strategy_feature_selection_variants(features, labels, cv):
    """Test different feature selection methods"""
    print("\n" + "="*70)
    print("STRATEGY 3: Feature Selection Variants")
    print("="*70)
    
    results = []
    
    # Best known RF params as baseline
    best_rf_params = {
        'n_estimators': 700,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'max_samples': 0.9,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Variant 1: No feature selection
    print("\n3.1: No Feature Selection")
    pipeline_no_fs = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('model', RandomForestClassifier(**best_rf_params))
    ])
    start = time.time()
    scores = cross_val_score(pipeline_no_fs, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
    results.append({
        'variant': 'No_FS',
        'cv_auc': scores.mean(),
        'cv_std': scores.std(),
        'time': time.time() - start
    })
    print(f"  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Variant 2: SelectFromModel with median threshold
    print("\n3.2: SelectFromModel (median threshold)")
    pipeline_sfm_median = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='median'
        )),
        ('model', RandomForestClassifier(**best_rf_params))
    ])
    start = time.time()
    scores = cross_val_score(pipeline_sfm_median, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
    results.append({
        'variant': 'SFM_median',
        'cv_auc': scores.mean(),
        'cv_std': scores.std(),
        'time': time.time() - start
    })
    print(f"  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Variant 3: SelectFromModel with mean threshold
    print("\n3.3: SelectFromModel (mean threshold)")
    pipeline_sfm_mean = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='mean'
        )),
        ('model', RandomForestClassifier(**best_rf_params))
    ])
    start = time.time()
    scores = cross_val_score(pipeline_sfm_mean, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
    results.append({
        'variant': 'SFM_mean',
        'cv_auc': scores.mean(),
        'cv_std': scores.std(),
        'time': time.time() - start
    })
    print(f"  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Variant 4: SelectKBest with f_classif
    print("\n3.4: SelectKBest (top 50% features)")
    n_features = features.shape[1]
    pipeline_kbest = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('feature_selection', SelectKBest(f_classif, k=max(10, n_features // 2))),
        ('model', RandomForestClassifier(**best_rf_params))
    ])
    start = time.time()
    scores = cross_val_score(pipeline_kbest, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
    results.append({
        'variant': 'KBest_50pct',
        'cv_auc': scores.mean(),
        'cv_std': scores.std(),
        'time': time.time() - start
    })
    print(f"  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Variant 5: More aggressive RF feature selector
    print("\n3.5: SelectFromModel with more estimators")
    pipeline_sfm_aggressive = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
            threshold='0.75*median'
        )),
        ('model', RandomForestClassifier(**best_rf_params))
    ])
    start = time.time()
    scores = cross_val_score(pipeline_sfm_aggressive, features, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
    results.append({
        'variant': 'SFM_aggressive',
        'cv_auc': scores.mean(),
        'cv_std': scores.std(),
        'time': time.time() - start
    })
    print(f"  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Find best variant
    best_idx = np.argmax([r['cv_auc'] for r in results])
    best_variant = results[best_idx]
    
    print(f"\n✓ Best variant: {best_variant['variant']}")
    print(f"  CV AUC: {best_variant['cv_auc']:.4f} ± {best_variant['cv_std']:.4f}")
    
    return {
        'name': f'RF_FS_{best_variant["variant"]}',
        'all_results': results,
        'best_variant': best_variant
    }


# Strategy 4: Fine-tuned optimization on best configuration
def strategy_fine_tuning(features, labels, cv, best_so_far):
    """Fine-tune around the best configuration found"""
    print("\n" + "="*70)
    print("STRATEGY 4: Fine-Tuning Best Configuration")
    print("="*70)
    print(f"Starting from: {best_so_far['name']} (CV AUC: {best_so_far['cv_auc']:.4f})")
    
    # Extract base parameters from best model
    if 'params' in best_so_far:
        base_params = best_so_far['params']
    else:
        # Use defaults if not available
        base_params = {
            'model__n_estimators': 700,
            'model__max_depth': 20,
            'model__min_samples_split': 5,
            'model__min_samples_leaf': 2,
            'model__max_features': 'sqrt',
            'model__max_samples': 0.9,
            'model__class_weight': 'balanced',
        }
    
    # Create narrow search space around best params
    param_grid = {}
    
    # Fine-tune n_estimators
    base_n_est = base_params.get('model__n_estimators', 700)
    param_grid['model__n_estimators'] = [
        max(300, base_n_est - 200),
        base_n_est,
        base_n_est + 300
    ]
    
    # Fine-tune max_depth
    base_depth = base_params.get('model__max_depth', 20)
    if base_depth is not None:
        param_grid['model__max_depth'] = [
            max(10, base_depth - 5),
            base_depth,
            base_depth + 5
        ]
    else:
        param_grid['model__max_depth'] = [20, 25, None]
    
    # Fine-tune other params
    param_grid['model__min_samples_split'] = [2, 5, 8]
    param_grid['model__min_samples_leaf'] = [1, 2, 4]
    param_grid['model__max_features'] = ['sqrt', 'log2', 0.3]
    
    # Build pipeline
    is_extratrees = 'Extra' in best_so_far['name']
    model_class = ExtraTreesClassifier if is_extratrees else RandomForestClassifier
    
    pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=-1, strategy='median')),
        ('scale', RobustScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            threshold='median'
        )),
        ('model', model_class(random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    
    print("Running GridSearchCV for fine-tuning...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    start = time.time()
    grid_search.fit(features, labels)
    elapsed = time.time() - start
    
    print(f"\n✓ Fine-tuning completed in {elapsed/60:.1f} minutes")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"Improvement: {grid_search.best_score_ - best_so_far['cv_auc']:+.4f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    return {
        'name': f'{best_so_far["name"]}_FineTuned',
        'pipeline': grid_search.best_estimator_,
        'cv_auc': grid_search.best_score_,
        'params': grid_search.best_params_,
        'search_time': elapsed
    }


def evaluate_final_models(strategies, features, labels, cv, output_folder):
    """Evaluate all strategies and pick the best"""
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    results = []
    
    for strategy in strategies:
        if 'pipeline' not in strategy:
            continue
            
        print(f"\nEvaluating {strategy['name']}...")
        
        # Cross-validation
        cv_scores = cross_val_score(
            strategy['pipeline'], features, labels,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        # Train on full data and get predictions for TPR calculation
        strategy['pipeline'].fit(features, labels)
        predictions = strategy['pipeline'].predict_proba(features)[:, 1]
        tpr, fpr_curve, tpr_curve = tprAtFPR(labels, predictions, desiredFPR=0.01)
        
        results.append({
            'name': strategy['name'],
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'tpr_at_fpr001': tpr,
            'pipeline': strategy['pipeline'],
            'params': strategy.get('params', {}),
            'fpr_curve': fpr_curve,
            'tpr_curve': tpr_curve
        })
        
        print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  TPR @ FPR=0.01: {tpr:.4f}")
    
    # Sort by CV AUC
    results.sort(key=lambda x: x['cv_auc_mean'], reverse=True)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY (sorted by CV AUC)")
    print("="*70)
    print(f"{'Rank':<6} {'Strategy':<30} {'CV AUC':<20} {'TPR@0.01':<10}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<6} {r['name']:<30} {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f}    {r['tpr_at_fpr001']:.4f}")
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Rank': i,
            'Strategy': r['name'],
            'CV_AUC_Mean': r['cv_auc_mean'],
            'CV_AUC_Std': r['cv_auc_std'],
            'TPR_at_FPR_001': r['tpr_at_fpr001']
        }
        for i, r in enumerate(results, 1)
    ])
    results_df.to_csv(os.path.join(output_folder, 'optimization_results.csv'), index=False)
    
    # Save best model
    best = results[0]
    with open(os.path.join(output_folder, 'best_rf_model.pkl'), 'wb') as f:
        pickle.dump(best['pipeline'], f)
    
    # Save best parameters
    with open(os.path.join(output_folder, 'best_parameters.txt'), 'w') as f:
        f.write(f"BEST RANDOM FOREST CONFIGURATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Strategy: {best['name']}\n")
        f.write(f"CV AUC: {best['cv_auc_mean']:.4f} ± {best['cv_auc_std']:.4f}\n")
        f.write(f"TPR @ FPR=0.01: {best['tpr_at_fpr001']:.4f}\n\n")
        f.write("Parameters:\n")
        for param, value in best['params'].items():
            f.write(f"  {param}: {value}\n")
    
    # Plot comparison
    plot_results_comparison(results, output_folder)
    
    return results


def plot_results_comparison(results, output_folder):
    """Create visualization of all results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. CV AUC comparison
    names = [r['name'] for r in results]
    aucs = [r['cv_auc_mean'] for r in results]
    stds = [r['cv_auc_std'] for r in results]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results)))
    bars = ax1.barh(range(len(names)), aucs, xerr=stds, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('CV AUC Score', fontsize=11)
    ax1.set_title('Cross-Validation AUC Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=max(aucs), color='red', linestyle='--', alpha=0.5, label='Best')
    ax1.legend()
    
    # Add value labels
    for i, (auc, std) in enumerate(zip(aucs, stds)):
        ax1.text(auc, i, f'  {auc:.4f}±{std:.4f}', va='center', fontsize=8)
    
    # 2. TPR comparison
    tprs = [r['tpr_at_fpr001'] for r in results]
    bars = ax2.barh(range(len(names)), tprs, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('TPR @ FPR=0.01', fontsize=11)
    ax2.set_title('True Positive Rate at FPR=0.01', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=max(tprs), color='red', linestyle='--', alpha=0.5, label='Best')
    ax2.legend()
    
    for i, tpr in enumerate(tprs):
        ax2.text(tpr, i, f'  {tpr:.4f}', va='center', fontsize=8)
    
    # 3. ROC Curves
    for i, r in enumerate(results[:5]):  # Top 5 only
        ax3.plot(r['fpr_curve'], r['tpr_curve'], 
                label=f"{r['name'][:25]} (AUC={r['cv_auc_mean']:.3f})",
                linewidth=2, alpha=0.8, color=colors[i])
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('True Positive Rate', fontsize=11)
    ax3.set_title('ROC Curves - Top 5 Strategies', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. ROC Curves - Zoomed (low FPR region)
    for i, r in enumerate(results[:5]):
        ax4.plot(r['fpr_curve'], r['tpr_curve'],
                label=f"{r['name'][:25]}",
                linewidth=2.5, alpha=0.8, color=colors[i])
    ax4.axvline(x=0.01, color='red', linestyle=':', alpha=0.5, label='FPR=0.01')
    ax4.set_xlim([0, 0.1])
    ax4.set_ylim([0, 0.8])
    ax4.set_xlabel('False Positive Rate', fontsize=11)
    ax4.set_ylabel('True Positive Rate', fontsize=11)
    ax4.set_title('ROC Curves - Low FPR Region', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'rf_optimization_results.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_folder}/rf_optimization_results.png")
    plt.close()


def main():
    print("="*70)
    print("RANDOM FOREST OPTIMIZATION SUITE")
    print("="*70)
    print("This script will test multiple strategies to maximize RF performance:")
    print("  1. Exhaustive hyperparameter search")
    print("  2. ExtraTrees comparison")
    print("  3. Feature selection variants")
    print("  4. Fine-tuning best configuration")
    print("="*70)
    
    # Create output folder
    output_folder = create_output_folder()
    print(f"\nOutput folder: {output_folder}")
    
    # Load data
    print("\nLoading data...")
    data = np.loadtxt('src/spamTrain1.csv', delimiter=',')
    features = data[:, :-1]
    labels = data[:, -1]
    print(f"Data shape: {features.shape}")
    print(f"Class distribution: {np.bincount(labels.astype(int))}")
    
    # Setup CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Run strategies
    all_strategies = []
    
    # Strategy 1: Exhaustive search
    strategy1 = strategy_random_forest_exhaustive(features, labels, cv)
    all_strategies.append(strategy1)
    
    # Strategy 2: ExtraTrees
    strategy2 = strategy_extra_trees(features, labels, cv)
    all_strategies.append(strategy2)
    
    # Strategy 3: Feature selection variants
    strategy3 = strategy_feature_selection_variants(features, labels, cv)
    # Note: strategy3 doesn't return a pipeline, just results
    
    # Find best so far for fine-tuning
    best_so_far = max(all_strategies, key=lambda x: x['cv_auc'])
    
    # Strategy 4: Fine-tuning
    strategy4 = strategy_fine_tuning(features, labels, cv, best_so_far)
    all_strategies.append(strategy4)
    
    # Final evaluation
    final_results = evaluate_final_models(all_strategies, features, labels, cv, output_folder)
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Results saved to: {output_folder}/")
    print(f"  - optimization_results.csv")
    print(f"  - best_rf_model.pkl")
    print(f"  - best_parameters.txt")
    print(f"  - rf_optimization_results.png")
    print("\nBest model:")
    print(f"  Strategy: {final_results[0]['name']}")
    print(f"  CV AUC: {final_results[0]['cv_auc_mean']:.4f} ± {final_results[0]['cv_auc_std']:.4f}")
    print(f"  TPR @ FPR=0.01: {final_results[0]['tpr_at_fpr001']:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

