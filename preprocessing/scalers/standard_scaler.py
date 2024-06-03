import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    This scaler standardizes each feature individually such that they have zero mean and unit variance.

    Attributes
    ----------
    mean_ : array-like of shape (n_features,)
        The mean value for each feature in the training data.
    std_ : array-like of shape (n_features,)
        The standard deviation for each feature in the training data.

    Methods
    -------
    fit(X, y=None)
        Compute the mean and standard deviation to be used for later scaling.
    transform(X)
        Perform standardization by centering and scaling.
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """
        Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        self : object
            Fitted StandardScaler model.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        if np.any(self.std_ == 0):
            raise ValueError("Standard deviation of one or more features is zero")
        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            The standardized input samples.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted yet. Call fit() before transform()")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            The standardized input samples.
        """
        return self.fit(X, y).transform(X)


