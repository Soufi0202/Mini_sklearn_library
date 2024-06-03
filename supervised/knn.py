import numpy as np
from supervised.base import BaseModel
from collections import Counter

class KNN(BaseModel):
    """
    k-Nearest Neighbors (KNN) classifier.

    Parameters:
        k (int): The number of nearest neighbors to consider.

    Attributes:
        k (int): The number of nearest neighbors to consider.
        X_train (array-like): The training input samples.
        y_train (array-like): The target values.
    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
        Fits the KNN classifier to the training data.

        Parameters:
            X (array-like): The training input samples.
            y (array-like): The target values.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts class labels for input samples.

        Parameters:
            X (array-like): The input samples.

        Returns:
            array: The predicted class labels.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predicts the class label for a single input sample.

        Parameters:
            x (array-like): A single input sample.

        Returns:
            int: The predicted class label.
        """
        # Compute distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
