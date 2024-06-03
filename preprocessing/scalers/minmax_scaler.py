import numpy as np

class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such that it is in the given range 
    on the training set, e.g., between zero and one.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    Attributes
    ----------
    min_ : array-like of shape (n_features,)
        Per feature minimum seen in the data.
    max_ : array-like of shape (n_features,)
        Per feature maximum seen in the data.

    Methods
    -------
    fit(X, y=None)
        Compute the minimum and maximum to be used for later scaling.
    transform(X)
        Scale features of X according to feature_range.
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        self : object
            Fitted MinMaxScaler model.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        if np.any(self.max_ == self.min_):
            raise ValueError("All features have the same value, division by zero would occur")
        return self

    def transform(self, X):
        """
        Scale features of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            The scaled input samples.
        """
        if self.min_ is None or self.max_ is None:
            raise ValueError("Scaler not fitted yet. Call fit() before transform()")
        X_std = (X - self.min_) / (self.max_ - self.min_ + 1e-8)  # Avoid division by zero
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled

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
            The scaled input samples.
        """
        return self.fit(X, y).transform(X)


