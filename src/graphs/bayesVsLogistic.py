import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
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
imputer = SimpleImputer(missing_values=-1, strategy="median")
train_features = imputer.fit_transform(train_features)
test_features = imputer.transform(test_features)

# Combine for cross-validation
features = np.vstack([train_features, test_features])
labels = np.hstack([train_labels, test_labels])

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "Bernoulli Naive Bayes": BernoulliNB(),
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("=" * 70)
print("LOGISTIC REGRESSION VS NAIVE BAYES COMPARISON")
print("=" * 70)

# Store results
results = {}

# Evaluate each model
for name, model in models.items():
    print(f"\n{name}:")

    # Cross-validation
    cv_scores = cross_val_score(model, features, labels, cv=cv, scoring="roc_auc")
    print(f"  CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Train and test
    model.fit(train_features, train_labels)
    test_probs = model.predict_proba(test_features)[:, 1]
    test_auc = roc_auc_score(test_labels, test_probs)
    print(f"  Test AUC: {test_auc:.4f}")

    # Store for plotting
    results[name] = {
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "test_auc": test_auc,
        "test_probs": test_probs,
    }

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: AUC comparison
ax = axes[0]
model_names = list(results.keys())
cv_means = [results[name]["cv_mean"] for name in model_names]
cv_stds = [results[name]["cv_std"] for name in model_names]
test_aucs = [results[name]["test_auc"] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

ax.bar(x - width / 2, cv_means, width, yerr=cv_stds, label="CV AUC", alpha=0.8)
ax.bar(x + width / 2, test_aucs, width, label="Test AUC", alpha=0.8)
ax.set_ylabel("AUC Score")
ax.set_title("Model Comparison: AUC Scores", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha="right")
ax.legend()
ax.grid(alpha=0.3, axis="y")

# Plot 2: ROC curves
ax = axes[1]
for name in model_names:
    probs = results[name]["test_probs"]
    fpr, tpr, _ = roc_curve(test_labels, probs)
    auc = results[name]["test_auc"]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("bayes_vs_logistic.png", dpi=300, bbox_inches="tight")
print(f"\n{'='*70}")
print("Plot saved as 'bayes_vs_logistic.png'")
print("=" * 70)
plt.show()
