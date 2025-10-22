import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load training data
train_data = np.loadtxt("splitTrainData.csv", delimiter=",")
train_features = train_data[:, :-1]
train_labels = train_data[:, -1]

# Load test data
test_data = np.loadtxt("splitTrainStartTest.csv", delimiter=",")
test_features = test_data[:, :-1]
test_labels = test_data[:, -1]

# Impute missing values
imputer = SimpleImputer(missing_values=-1, strategy='median')
train_features = imputer.fit_transform(train_features)
test_features = imputer.transform(test_features)

# Train models
# C is inverse regularization strength - higher C means less regularization
model_no_reg = LogisticRegression(penalty=None, max_iter=10000, random_state=1)
model_l1 = LogisticRegression(penalty='l1', C=12.0, solver='liblinear', max_iter=10000, random_state=1)

model_no_reg.fit(train_features, train_labels)
model_l1.fit(train_features, train_labels)

model_gaussian = GaussianNB()
model_bernoulli = BernoulliNB()
model_gaussian.fit(train_features, train_labels)
model_bernoulli.fit(train_features, train_labels)

# Predict without regularization
test_probs_no_reg = model_no_reg.predict_proba(test_features)[:, 1]
test_preds_no_reg = model_no_reg.predict(test_features)
auc_no_reg = roc_auc_score(test_labels, test_probs_no_reg)
acc_no_reg = accuracy_score(test_labels, test_preds_no_reg)

# Predict with L1
test_probs_l1 = model_l1.predict_proba(test_features)[:, 1]
test_preds_l1 = model_l1.predict(test_features)
auc_l1 = roc_auc_score(test_labels, test_probs_l1)
acc_l1 = accuracy_score(test_labels, test_preds_l1)

# Predict with Naive Bayes
test_probs_gaussian = model_gaussian.predict_proba(test_features)[:, 1]
test_preds_gaussian = model_gaussian.predict(test_features)
auc_gaussian = roc_auc_score(test_labels, test_probs_gaussian)
acc_gaussian = accuracy_score(test_labels, test_preds_gaussian)

test_probs_bernoulli = model_bernoulli.predict_proba(test_features)[:, 1]
test_preds_bernoulli = model_bernoulli.predict(test_features)
auc_bernoulli = roc_auc_score(test_labels, test_probs_bernoulli)
acc_bernoulli = accuracy_score(test_labels, test_preds_bernoulli)

print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"Logistic Regression (No Regularization):")
print(f"  Accuracy: {acc_no_reg:.4f}")
print(f"  AUC: {auc_no_reg:.4f}")
print(f"  Non-zero coefficients: {np.sum(model_no_reg.coef_ != 0)}/{len(model_no_reg.coef_[0])}")
print(f"\nLogistic Regression (L1, C=20.0):")
print(f"  Accuracy: {acc_l1:.4f}")
print(f"  AUC: {auc_l1:.4f}")
print(f"  Non-zero coefficients: {np.sum(model_l1.coef_ != 0)}/{len(model_l1.coef_[0])}")
print(f"\nGaussian Naive Bayes:")
print(f"  Accuracy: {acc_gaussian:.4f}")
print(f"  AUC: {auc_gaussian:.4f}")
print(f"\nBernoulli Naive Bayes:")
print(f"  Accuracy: {acc_bernoulli:.4f}")
print(f"  AUC: {auc_bernoulli:.4f}")
print("=" * 50)

# Plot ROC curves
fpr_no_reg, tpr_no_reg, _ = roc_curve(test_labels, test_probs_no_reg)
fpr_l1, tpr_l1, _ = roc_curve(test_labels, test_probs_l1)
fpr_gaussian, tpr_gaussian, _ = roc_curve(test_labels, test_probs_gaussian)
fpr_bernoulli, tpr_bernoulli, _ = roc_curve(test_labels, test_probs_bernoulli)

# Calculate partial AUC for zoomed region (FPR 0-0.2)
def partial_auc(fpr, tpr, max_fpr=0.2):
    """Calculate AUC only up to max_fpr"""
    indices = fpr <= max_fpr
    if not np.any(indices):
        return 0.0
    partial_fpr = fpr[indices]
    partial_tpr = tpr[indices]
    # Add the interpolated point at max_fpr
    if partial_fpr[-1] < max_fpr and len(fpr) > len(partial_fpr):
        idx = len(partial_fpr)
        if idx < len(fpr):
            # Linear interpolation
            slope = (tpr[idx] - partial_tpr[-1]) / (fpr[idx] - partial_fpr[-1])
            interp_tpr = partial_tpr[-1] + slope * (max_fpr - partial_fpr[-1])
            partial_fpr = np.append(partial_fpr, max_fpr)
            partial_tpr = np.append(partial_tpr, interp_tpr)
    return np.trapz(partial_tpr, partial_fpr)

partial_auc_no_reg = partial_auc(fpr_no_reg, tpr_no_reg)
partial_auc_l1 = partial_auc(fpr_l1, tpr_l1)
partial_auc_gaussian = partial_auc(fpr_gaussian, tpr_gaussian)
partial_auc_bernoulli = partial_auc(fpr_bernoulli, tpr_bernoulli)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full ROC curve
axes[0].plot(fpr_no_reg, tpr_no_reg, 'b-', linewidth=2, label=f'No Reg (AUC={auc_no_reg:.3f})')
axes[0].plot(fpr_l1, tpr_l1, 'r-', linewidth=2, label=f'L1 (AUC={auc_l1:.3f})')
axes[0].plot(fpr_gaussian, tpr_gaussian, 'g-', linewidth=2, label=f'Gaussian NB (AUC={auc_gaussian:.3f})')
axes[0].plot(fpr_bernoulli, tpr_bernoulli, 'm-', linewidth=2, label=f'Bernoulli NB (AUC={auc_bernoulli:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - Full', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Zoomed ROC curve (FPR 0-0.2) - showing partial AUC
axes[1].plot(fpr_no_reg, tpr_no_reg, 'b-', linewidth=2, label=f'No Reg (pAUC={partial_auc_no_reg:.4f})')
axes[1].plot(fpr_l1, tpr_l1, 'r-', linewidth=2, label=f'L1 (pAUC={partial_auc_l1:.4f})')
axes[1].plot(fpr_gaussian, tpr_gaussian, 'g-', linewidth=2, label=f'Gaussian NB (pAUC={partial_auc_gaussian:.4f})')
axes[1].plot(fpr_bernoulli, tpr_bernoulli, 'm-', linewidth=2, label=f'Bernoulli NB (pAUC={partial_auc_bernoulli:.4f})')
axes[1].set_xlim(0, 0.2)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve - Zoomed (FPR 0-0.2, Partial AUC)', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_l1_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'logistic_l1_comparison.png'")
plt.show()
