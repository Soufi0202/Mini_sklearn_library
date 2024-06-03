import numpy as np
from supervised.base import BaseModel

class LinearRegression(BaseModel):
    """
    Linear regression model using the Normal Equation.

    Attributes:
        coefficients_ (array-like): The coefficients (slopes) of the linear model.
        intercept_ (float): The intercept term of the linear model.
    """
    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fits the linear regression model to the training data.

        Parameters:
            X (array-like): The training input samples.
            y (array-like): The target values.
        """
        # Adding a column of ones for the intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Using the Normal Equation to find the best parameters
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coefficients_ = theta_best[1:]

    def predict(self, X):
        """
        Predicts target values for input samples.

        Parameters:
            X (array-like): The input samples.

        Returns:
            array: The predicted target values.
        """
        return np.dot(X, self.coefficients_) + self.intercept_

class LogisticRegression(BaseModel):
    """
    Logistic regression model using gradient descent.

    Parameters:
        learning_rate (float): The learning rate for gradient descent.
        iterations (int): The number of iterations for gradient descent.

    Attributes:
        learning_rate (float): The learning rate for gradient descent.
        iterations (int): The number of iterations for gradient descent.
        weights (array-like): The weights (coefficients) of the logistic model.
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def sigmoid(self, z):
        """
        Computes the sigmoid function.

        Parameters:
            z (array-like): The input values.

        Returns:
            array: The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fits the logistic regression model to the training data using gradient descent.

        Parameters:
            X (array-like): The training input samples.
            y (array-like): The target binary values.
        """
        # Initialize weights
        self.weights = np.zeros(X.shape[1] + 1)
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept term

        # Gradient descent
        for i in range(self.iterations):
            z = np.dot(X, self.weights)
            predictions = self.sigmoid(z)
            # Update weights
            gradient = np.dot(X.T, (predictions - y)) / y.size
            self.weights -= self.learning_rate * gradient

    def predict_proba(self, X):
        """
        Predicts class probabilities for input samples.

        Parameters:
            X (array-like): The input samples.

        Returns:
            array: The predicted probabilities of the positive class.
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept term
        z = np.dot(X, self.weights)
        return self.sigmoid(z)

    def predict(self, X):
        """
        Predicts class labels for input samples.

        Parameters:
            X (array-like): The input samples.

        Returns:
            array: The predicted class labels (binary).
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
