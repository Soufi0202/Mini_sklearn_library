import numpy as np

class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.

    This class transforms categorical features to a one-hot numeric array, which is 
    commonly used for machine learning models that require numeric input.

    Attributes
    ----------
    class_index_ : dict
        A dictionary mapping labels to their corresponding indices.

    Methods
    -------
    fit(X)
        Fit one-hot encoder by finding all unique labels in X.
    transform(X)
        Transform labels in X to a one-hot encoded array.
    fit_transform(X)
        Fit one-hot encoder and transform labels in one step.
    """

    def __init__(self):
        self.class_index_ = {}

    def fit(self, X):
        """
        Fit one-hot encoder by finding all unique labels in X.

        Parameters
        ----------
        X : list of list of str
            The input data containing categorical labels to be encoded.

        Returns
        -------
        self : OneHotEncoder
            Fitted one-hot encoder.
        """
        unique_classes = set(val for sublist in X for val in sublist)
        self.class_index_ = {label: idx for idx, label in enumerate(sorted(unique_classes))}
        return self

    def transform(self, X):
        """
        Transform labels in X to a one-hot encoded array.

        Parameters
        ----------
        X : list of list of str
            The input data containing categorical labels to be encoded.

        Returns
        -------
        np.array
            One-hot encoded array.
        """
        results = []
        for sublist in X:
            row = [0] * len(self.class_index_)
            for val in sublist:
                row[self.class_index_[val]] = 1
            results.append(row)
        return np.array(results)

    def fit_transform(self, X):
        """
        Fit one-hot encoder and transform labels in one step.

        Parameters
        ----------
        X : list of list of str
            The input data containing categorical labels to be encoded.

        Returns
        -------
        np.array
            One-hot encoded array.
        """
        return self.fit(X).transform(X)

