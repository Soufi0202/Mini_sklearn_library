import numpy as np
import itertools

class PolynomialFeatures:
    """
    Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations of the features 
    with degree less than or equal to the specified degree.

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.

    Methods
    -------
    fit(X, y=None)
        Fit to data; this method does not do anything as no fitting is required.
    transform(X)
        Transform data to polynomial features.
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        """
        Fit to data; this method does not do anything as no fitting is required.

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
        Transform data to polynomial features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_poly : array-like of shape (n_samples, n_output_features)
            The transformed input samples.
        """
        if len(X) == 0:
            raise ValueError("Input array X is empty")
        n_samples, n_features = X.shape
        features = [np.ones(n_samples)]
        for degree in range(1, self.degree + 1):
            for feature_indices in itertools.combinations_with_replacement(range(n_features), degree):
                features.append(np.prod(X[:, feature_indices], axis=1))
        return np.vstack(features).T

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
        X_poly : array-like of shape (n_samples, n_output_features)
            The transformed input samples.
        """
        return self.transform(X)


