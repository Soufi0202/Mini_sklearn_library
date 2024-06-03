import numpy as np

class MLP:
    """
    MLP (Multilayer Perceptron) is a class for a simple feedforward neural network with one hidden layer.

    Parameters
    ----------
    n_hidden : int, default=30
        The number of neurons in the hidden layer.
    learning_rate : float, default=0.01
        The learning rate for weight updates.
    n_iters : int, default=1000
        The number of iterations over the training data.

    Attributes
    ----------
    weights_input_hidden : ndarray of shape (n_features, n_hidden)
        Weights between input layer and hidden layer.
    weights_hidden_output : ndarray of shape (n_hidden, n_outputs)
        Weights between hidden layer and output layer.
    bias_hidden : ndarray of shape (1, n_hidden)
        Bias terms for hidden layer.
    bias_output : ndarray of shape (1, n_outputs)
        Bias terms for output layer.

    Methods
    -------
    fit(X, y)
        Train the MLP on the given dataset.
    predict(X)
        Predict class labels for samples in X.
    _sigmoid(x)
        Apply the sigmoid activation function.
    _sigmoid_derivative(x)
        Compute the derivative of the sigmoid function.
    _one_hot(y, n_outputs)
        Convert class labels to one-hot encoded format.
    """

    def __init__(self, n_hidden=30, learning_rate=0.01, n_iters=1000):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights_input_hidden = None
        self.weights_hidden_output = None
        self.bias_hidden = None
        self.bias_output = None

    def _sigmoid(self, x):
        """
        Apply the sigmoid activation function.

        Parameters
        ----------
        x : ndarray
            The input data.

        Returns
        -------
        output : ndarray
            The output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.

        Parameters
        ----------
        x : ndarray
            The input data.

        Returns
        -------
        output : ndarray
            The derivative of the sigmoid function.
        """
        return x * (1 - x)

    def fit(self, X, y):
        """
        Train the MLP on the given dataset.

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
        n_outputs = len(np.unique(y))

        self.weights_input_hidden = np.random.randn(n_features, self.n_hidden)
        self.weights_hidden_output = np.random.randn(self.n_hidden, n_outputs)
        self.bias_hidden = np.zeros((1, self.n_hidden))
        self.bias_output = np.zeros((1, n_outputs))

        y_onehot = self._one_hot(y, n_outputs)

        for _ in range(self.n_iters):
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self._sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = self._sigmoid(final_input)

            error = y_onehot - final_output
            d_output = error * self._sigmoid_derivative(final_output)

            error_hidden = d_output.dot(self.weights_hidden_output.T)
            d_hidden = error_hidden * self._sigmoid_derivative(hidden_output)

            self.weights_hidden_output += hidden_output.T.dot(d_output) * self.learning_rate
            self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
            self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

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
        y_pred : array of shape (n_samples,)
            The predicted class labels.
        """
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self._sigmoid(hidden_input)

        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self._sigmoid(final_input)

        return np.argmax(final_output, axis=1)

    def _one_hot(self, y, n_outputs):
        """
        Convert class labels to one-hot encoded format.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The class labels.
        n_outputs : int
            The number of unique classes.

        Returns
        -------
        onehot : ndarray of shape (n_samples, n_outputs)
            The one-hot encoded class labels.
        """
        onehot = np.zeros((y.shape[0], n_outputs))
        onehot[np.arange(y.shape[0]), y] = 1
        return onehot
