import numpy as np

class BaggingClassifier:
    """
    A Bagging classifier.

    This class implements a Bagging (Bootstrap Aggregating) classifier, which is an ensemble 
    meta-estimator that fits base models on random subsets of the original dataset and then 
    aggregates their individual predictions to form a final prediction.

    Parameters
    ----------
    base_model : object
        The base model to fit on random subsets of the dataset. Must have fit and predict methods.
    n_estimators : int, default=10
        The number of base models to fit.
    max_samples : float, default=1.0
        The proportion of the dataset to include in each random subset (bootstrap sample).

    Attributes
    ----------
    base_model : object
        The base model to fit on random subsets of the dataset.
    n_estimators : int
        The number of base models to fit.
    max_samples : float
        The proportion of the dataset to include in each random subset (bootstrap sample).
    models : list
        List of fitted base models.
    """

    def __init__(self, base_model, n_estimators=10, max_samples=1.0):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def _bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample from the dataset.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        
        Returns
        -------
        X_sample : array-like of shape (n_samples, n_features)
            The bootstrap sample of input samples.
        y_sample : array-like of shape (n_samples,)
            The bootstrap sample of target values.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Fit the Bagging classifier on the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        n_samples = int(self.max_samples * len(X))
        self.models = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            model = self.base_model()
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        """
        Predict class labels for the input samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        y_pred = [np.bincount(predictions[:, i]).argmax() for i in range(predictions.shape[1])]
        return np.array(y_pred)


