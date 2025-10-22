import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load training data
train_data = np.loadtxt("splitTrainData.csv", delimiter=",")
train_features = train_data[:, :-1]
train_labels = train_data[:, -1]

# Impute missing values
imputer = SimpleImputer(missing_values=-1, strategy='median')
train_features = imputer.fit_transform(train_features)

# Train L1 logistic regression
model_l1 = LogisticRegression(penalty='l1', C=20.0, solver='liblinear', max_iter=10000, random_state=1)
model_l1.fit(train_features, train_labels)

# Get features removed by L1 (coefficient = 0)
l1_coefficients = model_l1.coef_[0]
l1_removed = np.where(l1_coefficients == 0)[0]
l1_kept = np.where(l1_coefficients != 0)[0]

# Train Random Forest to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(train_features, train_labels)
feature_importances = rf.feature_importances_

# Sort features by importance and remove the same number as L1 removed
n_features_to_remove = len(l1_removed)
sorted_indices = np.argsort(feature_importances)  # ascending order
rf_removed = sorted_indices[:n_features_to_remove]  # least important features
rf_kept = sorted_indices[n_features_to_remove:]

# Calculate overlap
overlap_removed = np.intersect1d(l1_removed, rf_removed)
overlap_kept = np.intersect1d(l1_kept, rf_kept)

# Calculate percentages
pct_removed_overlap = 100 * len(overlap_removed) / len(l1_removed) if len(l1_removed) > 0 else 0
pct_kept_overlap = 100 * len(overlap_kept) / len(l1_kept) if len(l1_kept) > 0 else 0

print("=" * 70)
print("FEATURE SELECTION COMPARISON: L1 vs Random Forest")
print("=" * 70)
print(f"\nTotal features: {len(l1_coefficients)}")
print(f"\nL1 Regularization (C=20.0):")
print(f"  Features removed: {len(l1_removed)}")
print(f"  Features kept: {len(l1_kept)}")
print(f"  Removed feature indices: {sorted(l1_removed.tolist())}")

print(f"\nRandom Forest (removing {n_features_to_remove} least important):")
print(f"  Features removed: {len(rf_removed)}")
print(f"  Features kept: {len(rf_kept)}")
print(f"  Removed feature indices: {sorted(rf_removed.tolist())}")

print(f"\nOverlap Analysis:")
print(f"  Features removed by BOTH methods: {len(overlap_removed)}")
print(f"  Overlap percentage (of L1 removed): {pct_removed_overlap:.1f}%")
print(f"  Removed by both: {sorted(overlap_removed.tolist())}")

print(f"\n  Features KEPT by BOTH methods: {len(overlap_kept)}")
print(f"  Overlap percentage (of L1 kept): {pct_kept_overlap:.1f}%")

# Show features removed by one method but not the other
only_l1_removed = np.setdiff1d(l1_removed, rf_removed)
only_rf_removed = np.setdiff1d(rf_removed, l1_removed)

print(f"\n  Removed by L1 only: {sorted(only_l1_removed.tolist())}")
print(f"  Removed by RF only: {sorted(only_rf_removed.tolist())}")

# Show feature importances for features removed by L1
print(f"\n" + "=" * 70)
print("FEATURE IMPORTANCES (for features removed by L1)")
print("=" * 70)
for idx in sorted(l1_removed):
    importance = feature_importances[idx]
    removed_by_rf = "YES" if idx in rf_removed else "NO"
    print(f"Feature {idx:2d}: importance={importance:.6f}  (removed by RF: {removed_by_rf})")

print("\n" + "=" * 70)
