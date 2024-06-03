import numpy as np
from collections import Counter

class VotingClassifier:
    """
    VotingClassifier is an ensemble learning method that combines the predictions from multiple models 
    and returns the most common class (majority vote).

    Parameters
    ----------
    models : list of classifiers
        The list of classifiers to be used for voting.

    Methods
    -------
    fit(X, y)
        Fit all the models on the training data.
    predict(X)
        Predict class labels for the input samples by majority voting.
    """

    def __init__(self, models):
        """
        Initialize the VotingClassifier with the given models.

        Parameters
        ----------
        models : list of classifiers
            The list of classifiers to be used for voting.
        """
        self.models = models

    def fit(self, X, y):
        """
        Fit all the models on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels for the input samples by majority voting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted class labels.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        y_pred = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(predictions.shape[1])]
        return np.array(y_pred)
