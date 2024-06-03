import numpy as np

class KNNImputer:
    """
    Impute missing values using k-Nearest Neighbors.

    Each missing value is imputed using the mean value of the `n_neighbors` nearest neighbors found in the training set. 
    The nearest neighbors are computed based on the Euclidean distance metric.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.

    Attributes
    ----------
    train_data : array-like of shape (n_samples, n_features)
        The training data with no missing values.

    Methods
    -------
    fit(X, y=None)
        Fit the imputer on the training data.
    transform(X)
        Impute missing values in X using the fitted imputer.
    fit_transform(X, y=None)
        Fit the imputer on the training data and transform X in one step.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.train_data = None

    def fit(self, X, y=None):
        """
        Fit the imputer on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        self : object
            Fitted KNNImputer model.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        
        # Store non-missing parts of X to use for calculating nearest neighbors
        self.train_data = X[~np.isnan(X).any(axis=1)]
        if len(self.train_data) == 0:
            raise ValueError("No non-missing values in the training data")
        return self

    def transform(self, X):
        """
        Impute missing values in X using the fitted imputer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples with missing values.

        Returns
        -------
        X_out : array-like of shape (n_samples, n_features)
            The input samples with imputed values.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        
        X_out = np.copy(X)
        for i in range(X.shape[0]):
            if np.isnan(X[i]).any():
                # Compute distances to points with no missing values
                distances = np.sqrt(np.nansum((self.train_data - X[i])**2, axis=1))
                # Get indices of the closest points
                nearest_indices = np.argsort(distances)[:self.n_neighbors]
                # Check if all neighbors have missing values
                if np.isnan(self.train_data[nearest_indices]).any():
                    continue
                # Compute mean of the nearest neighbors ignoring nans
                neighbor_values = self.train_data[nearest_indices]
                imputed_values = np.nanmean(neighbor_values, axis=0)
                # Only fill missing values
                nan_mask = np.isnan(X[i])
                X_out[i, nan_mask] = imputed_values[nan_mask]
        return X_out

    def fit_transform(self, X, y=None):
        """
        Fit the imputer on the training data and transform X in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples with missing values.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        X_out : array-like of shape (n_samples, n_features)
            The input samples with imputed values.
        """
        return self.fit(X, y).transform(X)


