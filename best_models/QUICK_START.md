# Quick Start Guide

## ✅ What You Have

A clean, configurable spam classification testing suite with 4 models:

1. **`classifySpam_HGB.py`** - Histogram Gradient Boosting (Best for AUC)
2. **`classifySpam_RF.py`** - Random Forest (🏆 Overall Best)
3. **`classifySpam_KNN.py`** - K-Nearest Neighbors
4. **`classifySpam_LogReg.py`** - Logistic Regression

## 🚀 Quick Test

### Run Comprehensive Evaluation (ONE FILE - DOES EVERYTHING!)
```bash
cd best_models
python comprehensive_evaluation.py
```

This interactive script lets you:
1. **Choose preprocessing** - Feature selection, imputation, scaling
2. **Customize model parameters** - Adjust hyperparameters for each model
3. **Set number of runs** - Multiple runs for robust results
4. **Get everything**:
   - Cross-validation results
   - Test set performance
   - ROC curves (full + zoomed to low FPR)
   - Learning curves (train vs validation) for each model
   - Overfitting analysis
   - Complete comparison plots

## ⚙️ How to Configure

Open any model file (e.g., `classifySpam_RF.py`) and edit the top section:

```python
# ==================== CONFIGURATION ====================
USE_FEATURE_SELECTION = False  # Try True to enable feature selection
IMPUTATION_STRATEGY = 'median'  # Options: 'mean', 'median', 'knn', 'iterative'
SCALING_METHOD = 'none'          # Options: 'none', 'standard', 'robust', 'minmax'
# ======================================================
```

## 🎯 Current Best Results

With default settings:

| Model | Test AUC | TPR @ FPR=0.01 |
|-------|----------|----------------|
| **Random Forest** | **0.8985** | **0.4367** |
| HGB | 0.8928 | 0.3946 |
| LogReg | 0.8828 | 0.1807 |
| KNN | 0.6925 | 0.1175 |

## 🔧 To Improve TPR @ FPR=0.01

The True Positive Rate at FPR=0.01 is still low. Try:

### 1. Enable Feature Selection

Edit ALL model files and set:
```python
USE_FEATURE_SELECTION = True
```

Then run:
```bash
python evaluate_models.py
```

### 2. Try Different Imputation for LogReg/KNN

In `classifySpam_LogReg.py` and `classifySpam_KNN.py`:
```python
IMPUTATION_STRATEGY = 'iterative'  # or 'knn'
```

### 3. Experiment with Configurations

Each model file can have different settings. For example:

**For better KNN performance:**
```python
# In classifySpam_KNN.py
USE_FEATURE_SELECTION = True    # Reduce dimensionality
IMPUTATION_STRATEGY = 'knn'     # Better for KNN
SCALING_METHOD = 'robust'       # Handle outliers
```

## 📊 View Results

After running `evaluate_models.py`:
- Console shows detailed metrics
- `model_comparison.png` has visual comparison
- ROC curves showing performance at low FPR

## 💡 Pro Tips

1. **Imputation first**: All models impute missing values BEFORE other preprocessing
2. **Scaling matters**: LogReg and KNN NEED scaling, trees (HGB/RF) don't
3. **Feature selection**: Helps high-dimensional data, try it!
4. **Start simple**: Default settings are good, iterate from there

## 🔄 Iteration Workflow

```
1. Edit configurations in model files
2. Run: python evaluate_models.py
3. Check results
4. Adjust and repeat
```

## 📁 Files Structure

```
best_models/
├── classifySpam_HGB.py       # Configurable HGB model
├── classifySpam_RF.py        # Configurable RF model  
├── classifySpam_KNN.py       # Configurable KNN model
├── classifySpam_LogReg.py    # Configurable LogReg model
├── evaluate_models.py        # Compare all models
├── README.md                 # Detailed documentation
└── QUICK_START.md           # This file
```

## ❓ Need Help?

See `README.md` for detailed configuration guide and recommended settings per model.

