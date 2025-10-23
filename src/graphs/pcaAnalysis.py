import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = np.loadtxt("spamTrainData.csv", delimiter=",")
features = data[:, :-1]
labels = data[:, -1]

# Use Random Forest to get feature importances
print("=" * 70)
print("FEATURE SELECTION WITH RANDOM FOREST")
print("=" * 70)
rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
rf.fit(features, labels)

# Get feature importances and identify 10 least important features
importances = rf.feature_importances_
indices_sorted = np.argsort(importances)
drop_indices = indices_sorted[:10]  # 10 least important
keep_indices = indices_sorted[10:]  # Keep the rest

print(f"Original features: {features.shape[1]}")
print(f"\n10 Least important features to drop: {sorted(drop_indices.tolist())}")
print(f"Keeping {len(keep_indices)} features\n")

# Drop the 10 least important features
features_reduced = features[:, keep_indices]

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_reduced)

# Perform PCA
pca = PCA()
features_pca = pca.fit_transform(features_scaled)

# Calculate variance metrics
cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1

# Print variance information
print("=" * 70)
print("PCA ANALYSIS (AFTER DROPPING 10 FEATURES)")
print("=" * 70)
print(f"Features for PCA: {features_reduced.shape[1]}")
print(f"\nComponents for 95% variance: {n_components_95}")
print(f"First 3 PCs explain: {np.sum(pca.explained_variance_ratio_[:3]):.3f}")

# Create visualizations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Cumulative variance
axes[0].plot(range(1, len(cumsum_variance) + 1), cumsum_variance, "bo-")
axes[0].axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Cumulative Variance")
axes[0].set_title("Cumulative Variance Explained", fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: First two principal components
spam = labels == 1
axes[1].scatter(
    features_pca[~spam, 0],
    features_pca[~spam, 1],
    c="blue",
    alpha=0.5,
    s=10,
    label="Not Spam",
)
axes[1].scatter(
    features_pca[spam, 0], features_pca[spam, 1], c="red", alpha=0.5, s=10, label="Spam"
)
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})")
axes[1].set_title("First Two Principal Components", fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: PC2 vs PC3
axes[2].scatter(
    features_pca[~spam, 1],
    features_pca[~spam, 2],
    c="blue",
    alpha=0.5,
    s=10,
    label="Not Spam",
)
axes[2].scatter(
    features_pca[spam, 1], features_pca[spam, 2], c="red", alpha=0.5, s=10, label="Spam"
)
axes[2].set_xlabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})")
axes[2].set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.3f})")
axes[2].set_title("PC2 vs PC3", fontweight="bold")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pca_analysis_reduced.png", dpi=300, bbox_inches="tight")
print(f"\nPlot saved as 'pca_analysis_reduced.png'")
plt.show()

print("=" * 70)
