import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load training data
train_data = np.loadtxt("../spamTrain1.csv", delimiter=",")
train_features = train_data[:, :-1]
train_labels = train_data[:, -1]

# Load test data
test_data = np.loadtxt("../spamTrain2.csv", delimiter=",")
test_features = test_data[:, :-1]
test_labels = test_data[:, -1]

# Impute missing values
imputer = SimpleImputer(missing_values=-1, strategy='median')
train_features = imputer.fit_transform(train_features)
test_features = imputer.transform(test_features)

# Apply TF-IDF transformation
# Note: TF-IDF expects non-negative values, so we ensure all values are >= 0
train_features_nonneg = np.maximum(train_features, 0)
test_features_nonneg = np.maximum(test_features, 0)

tfidf = TfidfTransformer()
train_features_tfidf = tfidf.fit_transform(train_features_nonneg).toarray()
test_features_tfidf = tfidf.transform(test_features_nonneg).toarray()

# Train L1 models - without TF-IDF
model_l1_no_tfidf = LogisticRegression(penalty='l1', C=10.0, solver='liblinear', max_iter=10000, random_state=1)
model_l1_no_tfidf.fit(train_features, train_labels)

# Train L1 models - with TF-IDF
model_l1_with_tfidf = LogisticRegression(penalty='l1', C=10.0, solver='liblinear', max_iter=10000, random_state=1)
model_l1_with_tfidf.fit(train_features_tfidf, train_labels)

# Train Bernoulli Naive Bayes - without and with TF-IDF
model_bernoulli_no_tfidf = BernoulliNB()
model_bernoulli_with_tfidf = BernoulliNB()
model_bernoulli_no_tfidf.fit(train_features, train_labels)
model_bernoulli_with_tfidf.fit(train_features_tfidf, train_labels)

# Predict without TF-IDF
test_probs_no_tfidf = model_l1_no_tfidf.predict_proba(test_features)[:, 1]
test_preds_no_tfidf = model_l1_no_tfidf.predict(test_features)
auc_no_tfidf = roc_auc_score(test_labels, test_probs_no_tfidf)
acc_no_tfidf = accuracy_score(test_labels, test_preds_no_tfidf)

# Predict with TF-IDF
test_probs_with_tfidf = model_l1_with_tfidf.predict_proba(test_features_tfidf)[:, 1]
test_preds_with_tfidf = model_l1_with_tfidf.predict(test_features_tfidf)
auc_with_tfidf = roc_auc_score(test_labels, test_probs_with_tfidf)
acc_with_tfidf = accuracy_score(test_labels, test_preds_with_tfidf)

# Predict with Bernoulli NB - without TF-IDF
test_probs_bernoulli_no_tfidf = model_bernoulli_no_tfidf.predict_proba(test_features)[:, 1]
test_preds_bernoulli_no_tfidf = model_bernoulli_no_tfidf.predict(test_features)
auc_bernoulli_no_tfidf = roc_auc_score(test_labels, test_probs_bernoulli_no_tfidf)
acc_bernoulli_no_tfidf = accuracy_score(test_labels, test_preds_bernoulli_no_tfidf)

# Predict with Bernoulli NB - with TF-IDF
test_probs_bernoulli_with_tfidf = model_bernoulli_with_tfidf.predict_proba(test_features_tfidf)[:, 1]
test_preds_bernoulli_with_tfidf = model_bernoulli_with_tfidf.predict(test_features_tfidf)
auc_bernoulli_with_tfidf = roc_auc_score(test_labels, test_probs_bernoulli_with_tfidf)
acc_bernoulli_with_tfidf = accuracy_score(test_labels, test_preds_bernoulli_with_tfidf)

print("=" * 50)
print("MODEL COMPARISON: WITH vs WITHOUT TF-IDF")
print("=" * 50)
print(f"L1 without TF-IDF:")
print(f"  Accuracy: {acc_no_tfidf:.4f}")
print(f"  AUC: {auc_no_tfidf:.4f}")
print(f"  Non-zero coefficients: {np.sum(model_l1_no_tfidf.coef_ != 0)}/{len(model_l1_no_tfidf.coef_[0])}")
print(f"\nL1 with TF-IDF:")
print(f"  Accuracy: {acc_with_tfidf:.4f}")
print(f"  AUC: {auc_with_tfidf:.4f}")
print(f"  Non-zero coefficients: {np.sum(model_l1_with_tfidf.coef_ != 0)}/{len(model_l1_with_tfidf.coef_[0])}")
print(f"\nBernoulli NB without TF-IDF:")
print(f"  Accuracy: {acc_bernoulli_no_tfidf:.4f}")
print(f"  AUC: {auc_bernoulli_no_tfidf:.4f}")
print(f"\nBernoulli NB with TF-IDF:")
print(f"  Accuracy: {acc_bernoulli_with_tfidf:.4f}")
print(f"  AUC: {auc_bernoulli_with_tfidf:.4f}")
print("=" * 50)

# Plot ROC curves
fpr_no_tfidf, tpr_no_tfidf, _ = roc_curve(test_labels, test_probs_no_tfidf)
fpr_with_tfidf, tpr_with_tfidf, _ = roc_curve(test_labels, test_probs_with_tfidf)
fpr_bernoulli_no_tfidf, tpr_bernoulli_no_tfidf, _ = roc_curve(test_labels, test_probs_bernoulli_no_tfidf)
fpr_bernoulli_with_tfidf, tpr_bernoulli_with_tfidf, _ = roc_curve(test_labels, test_probs_bernoulli_with_tfidf)

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

partial_auc_no_tfidf = partial_auc(fpr_no_tfidf, tpr_no_tfidf)
partial_auc_with_tfidf = partial_auc(fpr_with_tfidf, tpr_with_tfidf)
partial_auc_bernoulli_no_tfidf = partial_auc(fpr_bernoulli_no_tfidf, tpr_bernoulli_no_tfidf)
partial_auc_bernoulli_with_tfidf = partial_auc(fpr_bernoulli_with_tfidf, tpr_bernoulli_with_tfidf)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full ROC curve
axes[0].plot(fpr_no_tfidf, tpr_no_tfidf, 'b-', linewidth=2, label=f'L1 without TF-IDF (AUC={auc_no_tfidf:.3f})')
axes[0].plot(fpr_with_tfidf, tpr_with_tfidf, 'r-', linewidth=2, label=f'L1 with TF-IDF (AUC={auc_with_tfidf:.3f})')
axes[0].plot(fpr_bernoulli_no_tfidf, tpr_bernoulli_no_tfidf, 'g-', linewidth=2, label=f'Bernoulli NB without TF-IDF (AUC={auc_bernoulli_no_tfidf:.3f})')
axes[0].plot(fpr_bernoulli_with_tfidf, tpr_bernoulli_with_tfidf, 'm-', linewidth=2, label=f'Bernoulli NB with TF-IDF (AUC={auc_bernoulli_with_tfidf:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - Full', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Zoomed ROC curve (FPR 0-0.2) - showing partial AUC
axes[1].plot(fpr_no_tfidf, tpr_no_tfidf, 'b-', linewidth=2, label=f'L1 without TF-IDF (pAUC={partial_auc_no_tfidf:.4f})')
axes[1].plot(fpr_with_tfidf, tpr_with_tfidf, 'r-', linewidth=2, label=f'L1 with TF-IDF (pAUC={partial_auc_with_tfidf:.4f})')
axes[1].plot(fpr_bernoulli_no_tfidf, tpr_bernoulli_no_tfidf, 'g-', linewidth=2, label=f'Bernoulli NB without TF-IDF (pAUC={partial_auc_bernoulli_no_tfidf:.4f})')
axes[1].plot(fpr_bernoulli_with_tfidf, tpr_bernoulli_with_tfidf, 'm-', linewidth=2, label=f'Bernoulli NB with TF-IDF (pAUC={partial_auc_bernoulli_with_tfidf:.4f})')
axes[1].set_xlim(0, 0.2)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve - Zoomed (FPR 0-0.2, Partial AUC)', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_tfidf_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'logistic_tfidf_comparison.png'")
plt.show()
