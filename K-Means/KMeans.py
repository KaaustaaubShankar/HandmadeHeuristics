import numpy as np

class KMeans:
    def __init__(self, clusters_count=8, tolerance=0.0001):
        self.clusters_count = clusters_count
        self.tolerance = tolerance
        self.centroids = []

    def initialize_centroids_random(self, data_points):
        random_indices = np.random.choice(data_points.shape[0], self.clusters_count, replace=False)
        centroids = data_points[random_indices]
        return centroids

    def train(self, inputs):
        # Initialize the centroids randomly
        self.centroids = self.initialize_centroids_random(inputs)

        while True:
            #calculate the new distance by finding mag^2 for each, sum, and then sqrt
            distances = np.sqrt(((inputs[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
            clusters = np.argmin(distances,axis=1)

            old_centroids = np.copy(self.centroids)

            #update the centroids according to the new clusters
            for k in range(self.clusters_count):
                #identify the points that make up cluster k
                pts_cluster = inputs[clusters == k]
                if len(pts_cluster) > 0:
                    self.centroids[k]= np.mean(pts_cluster,axis=0)
            
            #linalg.nrom calculates the size of the stuff within the array
            if (np.linalg.norm(self.centroids - old_centroids) <=self.tolerance):
                break

        return clusters
    
    def predict(self, data):
        distances = np.sqrt(((data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        print(distances)
        clusters = np.argmin(distances, axis=1)
        return clusters

