import numpy as np

class Perceptron:
    """
    Perceptron is a simple linear binary classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for updating the weights.
    n_iters : int, default=1000
        The number of iterations over the training data.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        The weights assigned to the features.
    bias : float
        The bias term.

    Methods
    -------
    fit(X, y)
        Fit the model to the training data.
    predict(X)
        Predict class labels for samples in X.
    _unit_step_func(x)
        Apply the unit step function to input x.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        """
        Apply the unit step function to input x.

        Parameters
        ----------
        x : ndarray
            The input data.

        Returns
        -------
        output : ndarray
            The binary output (0 or 1) after applying the unit step function.
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_predicted : array of shape (n_samples,)
            The predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
