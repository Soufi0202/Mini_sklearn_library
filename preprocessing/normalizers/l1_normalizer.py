import numpy as np

class L1Normalizer:
    """
    Normalize samples individually to have unit L1 norm.

    The L1 norm of each sample (row) is calculated, and each value in the sample 
    is divided by this norm to achieve a unit L1 norm.

    Methods
    -------
    fit(X, y=None)
        Fit to the data; this method does not do anything as no fitting is required.
    transform(X)
        Scale the input samples to have unit L1 norm.
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def fit(self, X, y=None):
        """
        Fit to the data; this method does not do anything as no fitting is required.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        self : object
            Returns self.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        return self

    def transform(self, X):
        """
        Scale the input samples to have unit L1 norm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_normalized : array-like of shape (n_samples, n_features)
            The normalized input samples.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        norms = np.sum(np.abs(X), axis=1, keepdims=True)
        return X / (norms + 1e-8)  # Avoid division by zero

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        X_normalized : array-like of shape (n_samples, n_features)
            The normalized input samples.
        """
        return self.transform(X)



