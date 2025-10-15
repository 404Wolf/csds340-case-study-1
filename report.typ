#set document(title: "Title", author: "Author")

#set page(
  paper: "us-letter",
  margin: 1in,
)

#set text(
  size: 12pt,
)

#align(center)[
  #text(size: 17pt, weight: "bold")[CSDS340 Case Study]

  *Wolf Mermelstein* and *Alessandro Mason* \
  Case Western Reserve University
]

#set heading(numbering: "1.")



= Feature Selection
Using a RF to find the features to use:

#figure(image("images/image.png"))



=== Comprehensive Model Comparison and Decision Process

To provide a thorough analysis of our spam classification problem, we conducted extensive experiments using three distinct approaches to systematically evaluate different preprocessing techniques and machine learning models. Our methodology was designed to understand not just which combinations work best, but why certain approaches are more effective for this specific dataset.

=== Methodology: Three-Tier Approach

**Approach 1: Original Single-Step Preprocessing**
We first tested the original approach where each preprocessing technique was applied individually, following the pattern from the initial `classifySpam.py`. This included testing all combinations of:
- Imputation strategies: mean, median, most frequent, KNN, iterative
- Scaling methods: standard, min-max, robust
- Feature selection: variance threshold, SelectKBest, L1 selection, PCA
- Models: Logistic Regression, Linear SVC, RBF SVC, Random Forest, HistGradientBoosting, KNN, Naive Bayes

**Approach 2: Random Forest-Based Feature Selection (MDI)**
Recognizing that Random Forest's Mean Decrease Impurity (MDI) provides excellent feature importance rankings, we used a trained Random Forest to identify the top 20 most important features from the original 30 features. This approach leverages the tree-based model's ability to capture non-linear feature interactions and provides a principled way to reduce dimensionality while preserving the most informative features.

**Approach 3: Combined Preprocessing Pipelines**
We implemented proper multi-step preprocessing pipelines that combine imputation, scaling, and feature selection in the correct order. This approach recognizes that preprocessing steps should be applied sequentially rather than as single-step transformations.

=== Results Analysis and Decision Process

**Key Finding 1: Model Performance Hierarchy**
Our comprehensive testing revealed a clear performance hierarchy that was consistent across all three approaches:

1. **HistGradientBoosting**: Consistently achieved the highest performance, with test AUC scores reaching 0.898
2. **Random Forest**: Strong performance with test AUC scores around 0.888-0.894
3. **KNN**: Moderate performance, particularly with robust scaling (0.697 test AUC)
4. **Linear Models**: Consistently poor performance across all approaches (0.494-0.518 test AUC)

**Key Finding 2: Feature Selection Impact**
The Random Forest-based feature selection (Approach 2) proved to be the most effective strategy:

- **HistGradientBoosting with RF-selected features**: Achieved the highest test AUC of 0.898 across multiple preprocessing combinations
- **Consistency**: The same top 20 features selected by Random Forest worked well with various preprocessing methods
- **Efficiency**: Reduced computational complexity while maintaining or improving performance

**Key Finding 3: Preprocessing Robustness**
Tree-based ensemble methods (Random Forest and HistGradientBoosting) showed remarkable robustness to different preprocessing approaches, while linear models consistently failed regardless of preprocessing strategy.

=== Final Model Selection and Rationale

**Selected Model: HistGradientBoosting with Random Forest Feature Selection**

**Rationale:**
1. **Highest Performance**: Achieved the best test AUC score of 0.898
2. **Consistency**: Performed well across multiple preprocessing combinations
3. **Feature Efficiency**: Benefits from the carefully selected 20 most important features
4. **Robustness**: Maintains performance with different imputation and scaling strategies
5. **Interpretability**: Feature importance from Random Forest provides insights into which features are most discriminative

**Preprocessing Pipeline:**
- **Feature Selection**: Top 20 features selected by Random Forest MDI
- **Imputation**: Iterative imputation (handles missing values more intelligently than simple strategies)
- **Scaling**: Robust scaling (less sensitive to outliers than standard scaling)
- **Model**: HistGradientBoosting with optimized hyperparameters

=== Performance Summary

| Approach | Best Model | Test AUC | CV Mean ± Std | Key Insight |
|----------|------------|----------|---------------|-------------|
| Original | SelectKBest + HistGradientBoosting | 0.897 | 0.905 ± 0.025 | Tree-based models dominate |
| RF Selection | RF-selected + HistGradientBoosting | 0.898 | 0.908 ± 0.024 | Feature selection improves performance |
| Combined | Robust + SelectKBest + HistGradientBoosting | 0.897 | 0.909 ± 0.022 | Proper preprocessing pipelines work well |

**Final Recommendation**: Use HistGradientBoosting with Random Forest-based feature selection, as it provides the best balance of performance, efficiency, and interpretability for this spam classification task.

=== Implementation Details

**Random Forest Feature Selection Implementation:**
````python
def select_features_with_rf(features, labels, n_features=20, random_state=42):
    """Use Random Forest to select most important features using MDI"""
    rf_selector = RandomForestClassifier(
        n_estimators=300, 
        class_weight="balanced_subsample", 
        n_jobs=-1, 
        random_state=random_state
    )
    rf_selector.fit(features, labels)
    
    # Get feature importance (MDI)
    importances = rf_selector.feature_importances_
    top_indices = np.argsort(importances)[-n_features:][::-1]
    
    return top_indices, importances
````

**Final Optimized Pipeline:**
````python
def create_optimized_pipeline():
    """Create the best performing pipeline"""
    return Pipeline([
        ('impute', IterativeImputer(random_state=0)),
        ('scale', RobustScaler()),
        ('model', HistGradientBoostingClassifier(
            random_state=0,
            learning_rate=0.1,
            max_depth=5,
            min_samples_leaf=20
        ))
    ])
````

**Key Technical Insights:**

1. **Feature Selection Strategy**: Random Forest's MDI proved more effective than mutual information or L1 regularization for this dataset, likely because it captures non-linear feature interactions that are crucial for spam detection.

2. **Preprocessing Order**: The correct sequence (imputation → scaling → feature selection → modeling) was critical for optimal performance.

3. **Model Robustness**: HistGradientBoosting's performance remained consistent across different preprocessing combinations, indicating its robustness to feature scaling and imputation strategies.

4. **Computational Efficiency**: Reducing from 30 to 20 features improved training time while maintaining or slightly improving performance, demonstrating the value of principled feature selection.

This comprehensive analysis demonstrates that systematic experimentation with multiple approaches, combined with proper understanding of preprocessing order and feature selection methods, leads to optimal model performance for spam classification tasks.

=== Advanced Random Forest Optimization: Challenging Feature Selection Assumptions

Following our initial findings that suggested feature selection improved model performance, we conducted a comprehensive Random Forest optimization study to maximize classification performance. This investigation led to a surprising and counterintuitive discovery that fundamentally challenged our earlier assumptions about feature selection.

**Optimization Methodology: Multi-Strategy Approach**

We implemented a systematic four-strategy optimization framework to explore the Random Forest parameter space thoroughly:

1. **Strategy 1 - Exhaustive Hyperparameter Search**: RandomizedSearchCV with 100 iterations across 10-fold stratified cross-validation, testing combinations of n_estimators (300-1000), max_depth (10-None), min_samples_split (2-15), min_samples_leaf (1-8), max_features ('sqrt', 'log2', 0.3-0.5), max_samples (0.7-1.0), criterion ('gini', 'entropy'), and class_weight settings. This comprehensive search identified optimal parameters: n_estimators=500, max_depth=10, max_features='log2', achieving CV AUC of 0.8977.

2. **Strategy 2 - ExtraTrees Investigation**: We tested ExtraTreesClassifier as an alternative, hypothesizing that its additional randomization (random thresholds rather than optimal thresholds) might reduce overfitting. Results confirmed this with CV AUC of 0.9022 and notably superior TPR at FPR=0.01 of 0.7279, demonstrating better performance in the critical low false-positive region.

3. **Strategy 3 - Feature Selection Ablation Study**: This was the most revealing component. We systematically compared five feature selection strategies against no feature selection:
   - No feature selection: **CV AUC 0.9130 ± 0.0230**
   - SelectFromModel (median threshold): CV AUC 0.8954 ± 0.0264
   - SelectFromModel (mean threshold): CV AUC 0.8913 ± 0.0255
   - SelectKBest (top 50%): CV AUC 0.9127 ± 0.0235
   - Aggressive SelectFromModel: CV AUC 0.9047 ± 0.0242

4. **Strategy 4 - Fine-tuning**: Focused GridSearchCV around the best configuration with 243 candidates, achieving final optimized performance.

**The Surprising Result: No Feature Selection Outperformed All Methods**

The most significant finding was that **removing feature selection entirely** yielded the best performance, with a 0.0176 AUC improvement over our previous best result (0.9130 vs 0.8954). This 1.86% improvement represents a substantial gain in classification accuracy. Our final optimized models achieved:

- **Optimized Random Forest**: CV AUC 0.9158 ± 0.0226, Test AUC 0.9006
- **ExtraTrees (best overall)**: CV AUC 0.9198 ± 0.0215, Test AUC 0.9068

Compared to our baseline RF with feature selection (CV AUC 0.895), this represents a **+2.48% improvement** to 0.9198, and a **+2.15% test improvement** from approximately 0.885 to 0.9068.

**Understanding the Paradox: When to Remove Features**

This finding appears to contradict conventional machine learning wisdom that feature selection reduces overfitting and improves generalization. However, several factors explain why retaining all features proved superior:

1. **Information Loss vs. Noise Reduction Trade-off**: While feature selection removes potentially noisy features, it also discards potentially informative features. In our 30-feature spam dataset, even features with lower individual importance may contain complementary information that, when aggregated across 500-700 trees, contributes meaningfully to classification accuracy. Tree-based ensemble methods are inherently robust to noisy features through bootstrapping and feature subsampling (max_features='log2' uses only ~1.6 features per split).

2. **Curse of Dimensionality Not Applicable**: The curse of dimensionality primarily affects distance-based methods (e.g., KNN) and linear models. With only 30 features and 1500 samples, our feature-to-sample ratio (1:50) is well within the range where tree-based ensembles excel. Random Forest's built-in feature subsampling effectively performs implicit feature selection at each split without permanently discarding features.

3. **Threshold Selection Sensitivity**: Feature selection methods require choosing importance thresholds (median, mean, etc.). Our results show high sensitivity to these choices—median threshold removed too many informative features (AUC 0.8954), while keeping all features allowed the ensemble to learn optimal feature combinations (AUC 0.9130). The "correct" threshold would need extensive tuning, negating the computational benefits of feature selection.

4. **Tree Ensemble Synergy**: Random Forests and ExtraTrees naturally handle feature redundancy through their ensemble structure. Each tree sees a random subset of features, so redundant features simply result in correlated trees, which the averaging process handles gracefully. In contrast, removing features based on single-tree importance (MDI) may eliminate features that become important in combination with others.

**Practical Guidelines for Feature Selection**

Our findings suggest a nuanced approach to feature selection:

**When to Apply Feature Selection:**
- High-dimensional data (p >> n, e.g., text classification with thousands of features)
- Computational constraints requiring faster training/prediction
- Feature acquisition costs make reducing features economically valuable
- Linear models or distance-based methods where curse of dimensionality applies
- Interpretability requirements mandate identifying minimal feature sets

**When to Retain All Features:**
- Moderate dimensionality (p < n/10) with ensemble methods
- Features are already curated/domain-selected (as in our spam dataset)
- Tree-based models with built-in feature subsampling
- Maximum accuracy is priority over interpretability or speed

**Optimized Model Specifications**

Our final recommended configuration:

**Random Forest Optimized:**
- n_estimators: 700 (increased from baseline 300)
- max_features: 'log2' (changed from 'sqrt')
- max_depth: 20 (relaxed from 15)
- min_samples_split: 5 (relaxed from 10)
- min_samples_leaf: 2 (relaxed from 5)
- max_samples: 0.9 (increased from 0.8)
- Preprocessing: Median imputation + RobustScaler
- No feature selection

**ExtraTrees Alternative (Highest Performance):**
- n_estimators: 500
- max_features: 'sqrt'
- max_depth: None (unrestricted)
- criterion: 'entropy'
- Superior TPR at FPR=0.01: 0.7279 vs RF's 0.6892

This optimization process consumed approximately 18 minutes of computation time across 1,223 model fits (100 + 80 randomized search iterations, 5 feature selection comparisons × 10 folds, 243 grid search candidates), demonstrating that systematic hyperparameter optimization yields substantial performance gains even for already-strong baseline models.
