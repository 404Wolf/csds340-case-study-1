# Random Forest Optimization Suite

## Overview
This script performs comprehensive optimization of Random Forest classifiers through multiple strategies to maximize performance on spam classification.

## What It Does

### Strategy 1: Exhaustive Hyperparameter Search
- Tests 100+ parameter combinations using RandomizedSearchCV
- Explores: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_samples, criterion, class_weight
- Uses robust scaling and RF-based feature selection

### Strategy 2: ExtraTrees Comparison
- Tests ExtraTreesClassifier (more randomization than standard RF)
- Can sometimes outperform RF by reducing overfitting
- 80 iterations of hyperparameter search

### Strategy 3: Feature Selection Variants
- Compares 5 different feature selection approaches:
  1. No feature selection
  2. SelectFromModel with median threshold
  3. SelectFromModel with mean threshold
  4. SelectKBest (top 50% features)
  5. Aggressive SelectFromModel (more estimators, stricter threshold)

### Strategy 4: Fine-Tuning
- Takes the best configuration from strategies 1-3
- Performs focused GridSearchCV around best parameters
- Final optimization pass

## Usage

### Run the optimization:
```bash
cd best_models
python3 optimize_rf.py
```

**Note:** This will take 30-60 minutes to complete, depending on your hardware.

## Output

All results are saved to a timestamped folder: `rf_optimization_YYYYMMDD_HHMMSS/`

### Files Created:
1. **`optimization_results.csv`** - Summary table of all strategies
2. **`best_rf_model.pkl`** - Serialized best model (can be loaded with pickle)
3. **`best_parameters.txt`** - Human-readable best configuration
4. **`rf_optimization_results.png`** - 4-panel visualization:
   - CV AUC comparison (all strategies)
   - TPR @ FPR=0.01 comparison
   - ROC curves (top 5 strategies)
   - Zoomed ROC curves (low FPR region)

## Expected Results

Based on current performance (~0.895 AUC), you can expect:
- **Baseline improvement**: +0.005 to +0.015 AUC through hyperparameter tuning
- **Feature selection optimization**: +0.002 to +0.008 AUC
- **ExtraTrees**: May match or slightly exceed RF performance
- **Total improvement**: Aiming for 0.900-0.910 AUC

## Loading the Best Model

```python
import pickle

# Load the best model
with open('rf_optimization_YYYYMMDD_HHMMSS/best_rf_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Use for predictions
predictions = best_model.predict_proba(test_features)[:, 1]
```

## Customization

To modify the search space, edit the `param_distributions` dictionaries in each strategy function.

To add more strategies, create a new function following the pattern of `strategy_*` functions and add it to the `main()` function.

## Progress Tracking

The script prints detailed progress:
- Current strategy being executed
- Search progress (verbose mode)
- Results after each strategy
- Final rankings and comparisons

## Tips

1. **Run overnight**: The optimization is thorough but time-consuming
2. **Monitor memory**: Multiple CV folds with large n_estimators can use significant RAM
3. **Save results**: All outputs are automatically saved—don't worry about interruptions
4. **Compare with baseline**: Check `comprehensive_evaluation.py` results for comparison

## What Makes This Optimization Effective

1. **Multi-pronged approach**: Tests fundamentally different strategies
2. **Smart search**: Uses RandomizedSearchCV for broad exploration, then GridSearchCV for refinement
3. **Practical metrics**: Optimizes for AUC but also tracks TPR @ FPR=0.01
4. **Feature selection**: Systematically tests whether FS helps or hurts
5. **ExtraTrees alternative**: Sometimes works better for high-dimensional data
6. **Reproducible**: Fixed random seeds ensure consistent results

## Next Steps After Optimization

1. Review `best_parameters.txt` to understand what worked
2. Update `comprehensive_evaluation.py` with the best parameters
3. Test the best model on held-out test data
4. Consider ensemble methods (combine best RF with HGB)

