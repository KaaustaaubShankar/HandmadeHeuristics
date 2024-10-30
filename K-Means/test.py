import numpy as np
import matplotlib.pyplot as plt
from KMeans import KMeans

# Step 1: Create synthetic data
# Generate three clusters of data points in 2D
np.random.seed(42)  # For reproducibility
data_cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
data_cluster2 = np.random.normal([8, 8], 0.5, (50, 2))
data_cluster3 = np.random.normal([5, 14], 0.5, (50, 2))

# Combine all data points into a single dataset
data_points = np.vstack((data_cluster1, data_cluster2, data_cluster3))

# Step 2: Instantiate and train the KMeans model
kmeans = KMeans(clusters_count=3, tolerance=0.0001)
cluster_assignments = kmeans.train(data_points)

# Step 3: Predict cluster for new data points
new_data = np.array([[1.5, 1.5], [8.5, 7.5], [5, 13.5]])
predicted_clusters = kmeans.predict(new_data)
print("Predicted clusters for new data:", predicted_clusters)

# Step 4: Visualization
# Plot the original clusters
plt.scatter(data_points[:, 0], data_points[:, 1], c=cluster_assignments, cmap='viridis', marker='o', label='Training Data')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=100, label='Centroids')

# Plot new data points
plt.scatter(new_data[:, 0], new_data[:, 1], c=predicted_clusters, cmap='viridis', marker='D', edgecolor='black', s=100, label='New Data')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('K-means Clustering with Predicted Data Points')
plt.show()
