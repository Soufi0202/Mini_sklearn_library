import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    This implementation assumes that the features are continuous and follow a Gaussian distribution.

    Attributes:
        classes (array-like): The unique classes present in the training data.
        mean (array-like): The mean of each feature for each class.
        var (array-like): The variance of each feature for each class.
        priors (array-like): The prior probabilities of each class.
    """
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Fits the Gaussian Naive Bayes classifier to the training data.

        Parameters:
            X (array-like): The training input samples.
            y (array-like): The target values.
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        """
        Predicts class labels for input samples.

        Parameters:
            X (array-like): The input samples.

        Returns:
            array: The predicted class labels.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predicts the class label for a single input sample.

        Parameters:
            x (array-like): A single input sample.

        Returns:
            int: The predicted class label.
        """
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Computes the probability density function (PDF) for the Gaussian distribution.

        Parameters:
            class_idx (int): The index of the class.
            x (array-like): The input sample.

        Returns:
            array: The PDF values.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
