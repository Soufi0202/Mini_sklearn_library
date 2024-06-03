import numpy as np

class AdaBoostClassifier:
    """
    An AdaBoost classifier.

    This class implements the AdaBoost (Adaptive Boosting) algorithm, which is an ensemble 
    method that combines multiple weak classifiers to create a strong classifier. The base 
    model is trained iteratively, with each subsequent model focusing more on the instances 
    that were misclassified by previous models.

    Parameters
    ----------
    base_model : object
        The base model to fit on random subsets of the dataset. Must have fit and predict methods.
    n_estimators : int, default=50
        The number of base models to fit.
    learning_rate : float, default=1.0
        The learning rate shrinks the contribution of each classifier by learning_rate. There is 
        a trade-off between learning_rate and n_estimators.

    Attributes
    ----------
    base_model : object
        The base model to fit on random subsets of the dataset.
    n_estimators : int
        The number of base models to fit.
    learning_rate : float
        The learning rate shrinks the contribution of each classifier by learning_rate.
    models : list
        List of fitted base models.
    model_weights : list
        List of weights for each model.
    """

    def __init__(self, base_model, n_estimators=50, learning_rate=1.0):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        """
        Fit the AdaBoost classifier on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples
        self.models = []
        self.model_weights = []

        for _ in range(self.n_estimators):
            model = self.base_model()
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            err = np.sum(w * (y_pred != y)) / np.sum(w)
            alpha = self.learning_rate * np.log((1 - err) / (err + 1e-10))
            w = w * np.exp(alpha * (y_pred != y))
            w = w / np.sum(w)

            self.models.append(model)
            self.model_weights.append(alpha)

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
        model_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.model_weights, model_preds)
        y_pred = np.sign(weighted_preds)
        return y_pred



import numpy as np

class GradientBoostingClassifier:
    """
    A Gradient Boosting classifier.

    This class implements the Gradient Boosting algorithm, which is an ensemble method that
    combines multiple weak classifiers to create a strong classifier. The base model is trained 
    iteratively, with each subsequent model focusing on the residual errors made by previous models.

    Parameters
    ----------
    base_model : object
        The base model to fit on random subsets of the dataset. Must have fit and predict methods.
    n_estimators : int, default=100
        The number of base models to fit.
    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each classifier by learning_rate. There is
        a trade-off between learning_rate and n_estimators.

    Attributes
    ----------
    base_model : object
        The base model to fit on random subsets of the dataset.
    n_estimators : int
        The number of base models to fit.
    learning_rate : float
        The learning rate shrinks the contribution of each classifier by learning_rate.
    models : list
        List of fitted base models.
    initial_prediction : float
        Initial prediction value based on the mean of the target values.
    """

    def __init__(self, base_model, n_estimators=100, learning_rate=0.1):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Fit the Gradient Boosting classifier on the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        n_samples, n_features = X.shape
        self.models = []

        # Initialize model with a constant prediction
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            
            model = self.base_model()
            model.fit(X, residuals)
            y_pred += self.learning_rate * model.predict(X)
            
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
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for model in self.models:
            y_pred += self.learning_rate * model.predict(X)
        return np.round(y_pred)



