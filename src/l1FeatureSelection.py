import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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

# Range of C values to test (log scale)
c_values = np.logspace(-2, 3, 50)  # From 0.01 to 1000

# Store results
features_removed = []
features_kept = []
auc_scores = []

print("Testing L1 regularization with different C values...")
print("=" * 60)

for c in c_values:
    # Train L1 model
    model = LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=10000, random_state=1)
    model.fit(train_features, train_labels)
    
    # Count non-zero coefficients
    n_nonzero = np.sum(model.coef_ != 0)
    n_zero = np.sum(model.coef_ == 0)
    
    # Calculate AUC on test set
    test_probs = model.predict_proba(test_features)[:, 1]
    auc = roc_auc_score(test_labels, test_probs)
    
    features_kept.append(n_nonzero)
    features_removed.append(n_zero)
    auc_scores.append(auc)

print(f"C range: {c_values[0]:.4f} to {c_values[-1]:.2f}")
print(f"Features removed range: {min(features_removed)} to {max(features_removed)}")
print(f"Features kept range: {min(features_kept)} to {max(features_kept)}")
print(f"AUC range: {min(auc_scores):.4f} to {max(auc_scores):.4f}")
print("=" * 60)

# ===== Random Forest Feature Importance Analysis =====
print("\n" + "=" * 60)
print("Testing Random Forest with different max_features settings...")
print("=" * 60)

# Range of max_features to test (as fraction of total features)
max_features_values = np.linspace(0.1, 1.0, 20)  # From 10% to 100% of features

# Store results for RF
rf_features_used = []
rf_auc_scores = []

for max_feat in max_features_values:
    # Train Random Forest with feature restriction
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_features=max_feat,
        random_state=1,
        n_jobs=-1
    )
    rf_model.fit(train_features, train_labels)
    
    # Get feature importances
    importances = rf_model.feature_importances_
    n_important = np.sum(importances > 0.01)  # Count features with >1% importance
    
    # Calculate AUC on test set
    test_probs = rf_model.predict_proba(test_features)[:, 1]
    auc = roc_auc_score(test_labels, test_probs)
    
    rf_features_used.append(n_important)
    rf_auc_scores.append(auc)

print(f"Max features range: {max_features_values[0]:.2f} to {max_features_values[-1]:.2f}")
print(f"Important features range: {min(rf_features_used)} to {max(rf_features_used)}")
print(f"AUC range: {min(rf_auc_scores):.4f} to {max(rf_auc_scores):.4f}")
print("=" * 60)

# Create 1x2 subplot layout for both L1 and RF
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 6))

total_features = train_features.shape[1]

# ===== L1 REGULARIZATION PLOT (LEFT) =====
# Plot features removed on left y-axis
color1 = 'tab:blue'
ax1.set_xlabel('C (Inverse Regularization Strength)', fontsize=12)
ax1.set_ylabel('Number of Features Removed', color=color1, fontsize=12)
ax1.semilogx(c_values, features_removed, color=color1, linewidth=2, marker='o', markersize=4, label='Features Removed')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(-1, total_features + 1)
ax1.grid(alpha=0.3)

# Add annotation for half features removed
half_removed_idx = np.argmin(np.abs(np.array(features_removed) - total_features/2))
ax1.axvline(x=c_values[half_removed_idx], color='gray', linestyle=':', alpha=0.4)

# Create second y-axis for AUC
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Test AUC Score', color=color2, fontsize=12)
ax2.semilogx(c_values, auc_scores, color=color2, linewidth=2, marker='s', markersize=4, label='Test AUC')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(min(auc_scores) - 0.01, max(auc_scores) + 0.01)

# Find and mark best AUC
best_auc_idx = np.argmax(auc_scores)
ax1.axvline(x=c_values[best_auc_idx], color='red', linestyle='--', alpha=0.4)

# Add title
ax1.set_title('L1 Regularization: Feature Selection vs Performance', fontweight='bold', fontsize=14, pad=20)

# Add combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Add annotations
ax1.text(c_values[half_removed_idx], total_features * 0.95, 
        f'50% removed\nC={c_values[half_removed_idx]:.2f}', 
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax2.text(c_values[best_auc_idx], max(auc_scores), 
        f'Best AUC={auc_scores[best_auc_idx]:.4f}\nC={c_values[best_auc_idx]:.2f}', 
        ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# ===== RANDOM FOREST PLOT (RIGHT) =====
# Plot features retained on left y-axis
color3 = 'tab:blue'
ax3.set_xlabel('Max Features (Fraction of Total)', fontsize=12)
ax3.set_ylabel('Number of Important Features (>1% importance)', color=color3, fontsize=12)
ax3.plot(max_features_values, rf_features_used, color=color3, linewidth=2, marker='o', markersize=4, label='Important Features')
ax3.tick_params(axis='y', labelcolor=color3)
ax3.set_ylim(-1, total_features + 1)
ax3.grid(alpha=0.3)

# Create second y-axis for AUC
ax4 = ax3.twinx()
color4 = 'tab:green'
ax4.set_ylabel('Test AUC Score', color=color4, fontsize=12)
ax4.plot(max_features_values, rf_auc_scores, color=color4, linewidth=2, marker='s', markersize=4, label='Test AUC')
ax4.tick_params(axis='y', labelcolor=color4)
ax4.set_ylim(min(rf_auc_scores) - 0.01, max(rf_auc_scores) + 0.01)

# Find and mark best AUC
best_rf_auc_idx = np.argmax(rf_auc_scores)
ax3.axvline(x=max_features_values[best_rf_auc_idx], color='red', linestyle='--', alpha=0.4)

# Add title
ax3.set_title('Random Forest: Feature Usage vs Performance', fontweight='bold', fontsize=14, pad=20)

# Add combined legend
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=10)

# Add annotation for best AUC
ax4.text(max_features_values[best_rf_auc_idx], rf_auc_scores[best_rf_auc_idx], 
        f'Best AUC={rf_auc_scores[best_rf_auc_idx]:.4f}\nmax_features={max_features_values[best_rf_auc_idx]:.2f}', 
        ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Add overall figure note
fig.text(0.5, 0.02, 
         'Note: Left - L1 regularization (smaller C = stronger reg) | Right - RF feature usage (>1% importance)',
         ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('feature_selection_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'feature_selection_comparison.png'")
plt.show()
