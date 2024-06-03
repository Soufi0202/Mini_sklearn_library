import numpy as np

class MaxAbsScaler:
    """
    Scale each feature by its maximum absolute value.

    This scaler scales each feature individually such that the maximum absolute value 
    of each feature in the training set will be 1.0. It does not shift/center the data, 
    and thus does not destroy any sparsity.

    Attributes
    ----------
    max_abs_ : array-like of shape (n_features,)
        The maximum absolute value for each feature in the training data.

    Methods
    -------
    fit(X, y=None)
        Compute the maximum absolute value for each feature.
    transform(X)
        Scale the features of X by the computed maximum absolute values.
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def __init__(self):
        self.max_abs_ = None

    def fit(self, X, y=None):
        """
        Compute the maximum absolute value for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        self : object
            Fitted MaxAbsScaler model.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        self.max_abs_ = np.max(np.abs(X), axis=0)
        if np.any(self.max_abs_ == 0):
            raise ValueError("Max absolute value of one or more features is zero")
        return self

    def transform(self, X):
        """
        Scale the features of X by the computed maximum absolute values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_scaled : array-like of shape (n_samples, n_features)
            The scaled input samples.
        """
        if self.max_abs_ is None:
            raise ValueError("Scaler not fitted yet. Call fit() before transform()")
        return X / self.max_abs_

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


