import numpy as np

class VarianceThreshold:
    """
    Feature selector that removes all low-variance features.

    This feature selection method removes all features whose variance does not meet 
    a specified threshold. By default, it removes all features with zero variance, 
    i.e., features that have the same value in all samples.

    Parameters
    ----------
    threshold : float, default=0.0
        Features with a training-set variance lower than this threshold will be removed.

    Attributes
    ----------
    variances_ : array-like, shape (n_features,)
        Variances of each feature.

    Methods
    -------
    fit(X, y=None)
        Compute the variances of the features.
    transform(X)
        Reduce X to the features with variance above the threshold.
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Compute the variances of the features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        self : object
            Fitted VarianceThreshold model.
        """
        self.variances = np.var(X, axis=0)
        return self

    def transform(self, X):
        """
        Reduce X to the features with variance above the threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_reduced : array-like of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        return X[:, self.variances > self.threshold]

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
        X_reduced : array-like of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        return self.fit(X, y).transform(X)





