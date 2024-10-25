import numpy as np
import pandas as pd

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two points.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X, task='classification'):
        """
        Predict the labels for the given data points.
        """
        X = np.array(X)
        predictions = [self._predict_single(x, task) for x in X]
        return np.array(predictions)

    def _predict_single(self, x, task):
        """
        Predict the label for a single data point.
        """
        # Calculate the distance from the new point to all training points
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Perform classification or regression
        if task == 'classification':
            # Return the most common class label
            return pd.Series(k_nearest_labels).mode()[0]
        elif task == 'regression':
            # Return the average of the nearest neighbor labels
            return np.mean(k_nearest_labels)
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")

