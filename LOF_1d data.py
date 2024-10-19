import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Generate sample 1D data (mostly normal points + some outliers)
data = np.array([1, 2, 15, 3, 4, 5, 6, 7, 2, 21]).reshape(-1, 1)

# Apply LOF for outlier detection
lof = LocalOutlierFactor(n_neighbors=2)  # Set neighbors to control sensitivity
outlier_labels = lof.fit_predict(data)  # -1 indicates outliers, 1 indicates inliers
lof_scores = -lof.negative_outlier_factor_  # LOF scores (higher = more likely outlier)

# Print the results
print(f"Labels: {outlier_labels}")
print(f"LOF Scores: {lof_scores}")

# Plot the data with outliers highlighted
plt.scatter(range(len(data)), data, c=outlier_labels, cmap='coolwarm', s=100)
plt.colorbar(label='Inlier (1) / Outlier (-1)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('1D Data Outlier Detection using LOF')
plt.show()
