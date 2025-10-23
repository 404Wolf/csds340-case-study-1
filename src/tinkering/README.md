# Best Models for Spam Classification

Clean, configurable implementation of the top-performing spam classification models.

## 📁 Models Included

1. **`classifySpam_HGB.py`** - Histogram Gradient Boosting
2. **`classifySpam_RF.py`** - Random Forest
3. **`classifySpam_KNN.py`** - K-Nearest Neighbors
4. **`classifySpam_LogReg.py`** - Logistic Regression

## 🎛️ Configuration

Each model file has configurable settings at the top:

```python
# ==================== CONFIGURATION ====================
USE_FEATURE_SELECTION = False  # Enable/disable RF-based feature selection
IMPUTATION_STRATEGY = 'median'  # Options: 'mean', 'median', 'knn', 'iterative'
SCALING_METHOD = 'none'          # Options: 'none', 'standard', 'robust', 'minmax'
# ======================================================
```

### Imputation Strategies

- **`'mean'`** - Replace missing values (-1) with feature mean
- **`'median'`** - Replace missing values with feature median (default for tree-based)
- **`'knn'`** - Use K-Nearest Neighbors to impute missing values
- **`'iterative'`** - Use iterative imputation (MICE algorithm)

### Scaling Methods

- **`'none'`** - No scaling (recommended for tree-based models: HGB, RF)
- **`'standard'`** - StandardScaler (mean=0, std=1) (recommended for LogReg)
- **`'robust'`** - RobustScaler (uses median and IQR) (recommended for KNN)
- **`'minmax'`** - MinMaxScaler (scales to [0,1] range)

### Feature Selection

- **`False`** - Use all features (after variance thresholding)
- **`True`** - Use Random Forest to select top 50% most important features

## 🚀 Usage

### **MAIN EVALUATION SCRIPT** (Recommended)

```bash
cd best_models
python comprehensive_evaluation.py
```

**This ONE file does EVERYTHING:**

- ✅ Interactive configuration (feature selection, imputation, scaling)
- ✅ Custom model parameters
- ✅ Multiple runs for robustness
- ✅ Cross-validation + test set evaluation
- ✅ Learning curves (train vs validation)
- ✅ Overfitting detection
- ✅ Complete comparison plots

See `USAGE_EXAMPLE.md` for detailed walkthrough.

### Run Individual Models (for quick tests)

```bash
python classifySpam_HGB.py      # Histogram Gradient Boosting
python classifySpam_RF.py       # Random Forest
python classifySpam_KNN.py      # K-Nearest Neighbors
python classifySpam_LogReg.py   # Logistic Regression
```

## 📊 Expected Performance (Default Settings)

| Model         | Test AUC | TPR @ FPR=0.01 |
| ------------- | -------- | -------------- |
| Random Forest | 0.8985   | 0.4367         |
| Histogram GB  | 0.8928   | 0.3946         |
| KNN           | 0.7834   | 0.0783         |
| Logistic Reg  | ~0.50    | ~0.00          |

## 🔬 Experimentation Guide

### To Improve TPR @ FPR=0.01:

1. **Enable Feature Selection**:

   ```python
   USE_FEATURE_SELECTION = True
   ```

2. **Try Different Imputation** (especially for KNN and LogReg):

   ```python
   IMPUTATION_STRATEGY = 'iterative'
   ```

3. **Experiment with Scaling** (important for KNN/LogReg):
   ```python
   SCALING_METHOD = 'robust'  # or 'standard' or 'minmax'
   ```

### Recommended Configurations by Model:

**HGB (Histogram Gradient Boosting)**:

```python
USE_FEATURE_SELECTION = False
IMPUTATION_STRATEGY = 'median'
SCALING_METHOD = 'none'
```

**Random Forest**:

```python
USE_FEATURE_SELECTION = False  # Try True if overfitting
IMPUTATION_STRATEGY = 'median'
SCALING_METHOD = 'none'
```

**KNN**:

```python
USE_FEATURE_SELECTION = True   # Helps with curse of dimensionality
IMPUTATION_STRATEGY = 'median'
SCALING_METHOD = 'robust'      # CRITICAL for distance-based methods
```

**Logistic Regression**:

```python
USE_FEATURE_SELECTION = True   # Helps with high-dimensional data
IMPUTATION_STRATEGY = 'median'
SCALING_METHOD = 'standard'    # CRITICAL for linear models
```

## 📝 Notes

- **Imputation is done FIRST** before any other preprocessing
- All models use stratified 10-fold cross-validation
- Test set uses odd/even split (odd samples = train, even = test)
- All models include regularization to reduce overfitting
- Random state is fixed at 42 for reproducibility

## 🎯 Next Steps

1. Edit configuration in model files
2. Run `evaluate_models.py` to compare
3. Check `model_comparison.png` for visual comparison
4. Iterate on best performing configuration
