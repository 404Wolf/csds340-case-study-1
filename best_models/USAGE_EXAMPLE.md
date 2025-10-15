# Usage Example for comprehensive_evaluation.py

## What You Get

**ONE FILE THAT DOES EVERYTHING:**
- Interactive configuration selection
- Custom model parameters
- Complete evaluation with multiple runs
- Learning curves showing train vs validation
- Overfitting detection
- Full comparison plots

## Quick Run

```bash
cd best_models
python comprehensive_evaluation.py
```

## Example Session

```
STEP 1: PREPROCESSING CONFIGURATION
1. Feature Selection:
   [1] Without feature selection
   [2] With RF-based feature selection (top 50%)
Choose (1 or 2): 2

2. Imputation Strategy:
   [1] Mean
   [2] Median (recommended for trees)
   [3] KNN
   [4] Iterative (MICE)
Choose (1-4): 2

3. Scaling Method:
   [1] None (for tree-based)
   [2] Standard (mean=0, std=1)
   [3] Robust (median/IQR)
   [4] MinMax ([0,1])
Choose (1-4): 3

4. Number of runs per model (for robustness):
Enter (default 3): 5

STEP 2: MODEL PARAMETERS
Customize model parameters? (y/n) [n]: y

--- HGB Parameters ---
  Learning rate [default: 0.05]: 0.1
  Max depth [default: 5]: 7
  Min samples leaf [default: 50]: 30
  L2 regularization [default: 1.0]: 

--- RF Parameters ---
  Number of trees [default: 300]: 500
  Max depth [default: 15, None for unlimited]: 20
  Min samples split [default: 10]: 
  Min samples leaf [default: 5]: 3

[... continues for KNN and LogReg ...]
```

## What Gets Generated

After running, you'll get:

### Files Created:
1. `comparison_*.png` - ROC curves (full + zoomed) + metrics bars
2. `learning_curve_hgb.png` - HGB train vs validation with overfitting analysis
3. `learning_curve_rf.png` - RF train vs validation with overfitting analysis  
4. `learning_curve_knn.png` - KNN train vs validation with overfitting analysis
5. `learning_curve_logreg.png` - LogReg train vs validation with overfitting analysis
6. `overfitting_summary.png` - Bar chart comparing overfitting across all models

### Console Output:
```
RESULTS SUMMARY
================================================================================
Model      CV AUC                 Test AUC               TPR@0.01            
--------------------------------------------------------------------------------
RF         0.9120 ± 0.0235     0.9015 ± 0.0023     0.4521 ± 0.0102     
HGB        0.9098 ± 0.0251     0.8975 ± 0.0018     0.4105 ± 0.0089     
LogReg     0.8995 ± 0.0268     0.8842 ± 0.0031     0.2134 ± 0.0156     
KNN        0.7234 ± 0.0312     0.7145 ± 0.0042     0.1387 ± 0.0201     

================================================================================
🏆 BEST: RF (AUC: 0.9015)
================================================================================

OVERFITTING SUMMARY (from learning curves):
Model                          Train-Val Gap        Status              
--------------------------------------------------------------------------------
LogReg                         0.0145               ✓ Good fit
HGB                            0.0298               ✓ Good fit
RF                             0.0412               ✓ Good fit
KNN                            0.2856               ✗ Significant overfitting
```

## Understanding Learning Curves

Each learning curve plot has **2 panels**:

### Left Panel: Learning Curve
- **Red line**: Training AUC (how well model fits training data)
- **Green line**: Validation AUC (how well model generalizes)
- Shows both increasing with more data
- Validation should be close to training

### Right Panel: Overfitting Gap
- **Orange area**: Difference between train and validation AUC
- **Green dashed line**: Good fit threshold (gap < 0.05)
- **Red dashed line**: Overfitting threshold (gap > 0.15)
- **Text box**: Final gap and status

### Interpretation:
- **Gap < 0.05**: ✓ Model generalizes well
- **Gap 0.05-0.15**: ⚠ Some overfitting but acceptable  
- **Gap > 0.15**: ✗ Significant overfitting - needs more regularization

## Tips

1. **Start simple**: Use defaults first to get baseline
2. **Enable feature selection** if you see overfitting
3. **Increase regularization** (lower learning rate, higher min_samples) if gap is large
4. **Run 3-5 times** for robust estimates (accounts for random variation)
5. **Scaling matters**: Always use for KNN/LogReg, optional for trees

## Quick Experiments

### Reduce Overfitting:
- Enable feature selection (option 2)
- Use robust scaling (option 3)
- Increase min_samples_leaf for trees
- Increase n_neighbors for KNN
- Decrease C (more regularization) for LogReg

### Improve TPR @ FPR=0.01:
- Enable feature selection
- Try iterative imputation
- Customize model parameters
- Run with more iterations/trees

