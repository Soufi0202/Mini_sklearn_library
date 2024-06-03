import numpy as np

class RFE:
    """
    Recursive Feature Elimination (RFE).

    RFE is a feature selection method that fits a model and removes the weakest features until 
    the desired number of features is reached. The model is trained repeatedly, with the 
    weakest features being removed each time.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that provides information about feature importance 
        (e.g., `feature_importances_` attribute).
    n_features_to_select : int, default=None
        The number of features to select. If None, half of the features are selected.

    Attributes
    ----------
    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ranking_[i] corresponds to the ranking position of the i-th feature.
    support_ : ndarray of shape (n_features,)
        The mask of selected features.
    """

    def __init__(self, estimator, n_features_to_select=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        """
        Fit the RFE model and rank features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted RFE model.
        """
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            self.n_features_to_select = n_features // 2

        self.ranking_ = np.ones(n_features, dtype=int)
        self.support_ = np.zeros(n_features, dtype=bool)

        while np.sum(self.support_) < self.n_features_to_select:
            self.estimator.fit(X[:, self.ranking_ == 1], y)
            importances = self.estimator.feature_importances_
            least_important = np.argsort(importances)[:1]
            self.ranking_[self.ranking_ == 1][least_important] += 1
            self.support_ = self.ranking_ == 1

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_reduced : array-like of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        return X[:, self.support_]

    def fit_transform(self, X, y):
        """
        Fit the RFE model and reduce X to the selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        X_reduced : array-like of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        return self.fit(X, y).transform(X)

class SimpleTreeEstimator:
    """
    A simple tree-based estimator for feature importance.

    This is a placeholder estimator that provides random feature importances. It is used for 
    demonstration purposes.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances (randomly generated).
    """

    def __init__(self):
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        n_samples, n_features = X.shape
        self.feature_importances_ = np.random.rand(n_features)

    def predict(self, X):
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        return np.random.randint(0, 2, size=(X.shape[0],))


