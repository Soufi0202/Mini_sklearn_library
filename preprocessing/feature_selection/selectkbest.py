import numpy as np

class SelectKBest:
    """
    Select features according to the k highest scores.

    SelectKBest is a filter method for feature selection that selects the top k features 
    based on a scoring function. The scoring function evaluates the relationship between 
    each feature and the target variable.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning an array of scores for each feature.
    k : int, default=10
        Number of top features to select.

    Attributes
    ----------
    scores_ : array-like, shape (n_features,)
        Scores of features.
    indices_ : array-like, shape (k,)
        Indices of the top k selected features.

    Methods
    -------
    fit(X, y)
        Fit the SelectKBest model according to the given data.
    transform(X)
        Reduce X to the selected features.
    fit_transform(X, y)
        Fit the SelectKBest model and transform X in one step.
    """

    def __init__(self, score_func, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        """
        Fit the SelectKBest model according to the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted SelectKBest model.
        """
        self.scores = self.score_func(X, y)
        self.indices = np.argsort(self.scores)[-self.k:]
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
        X_reduced : array-like of shape (n_samples, k)
            The input samples with only the selected features.
        """
        return X[:, self.indices]

    def fit_transform(self, X, y):
        """
        Fit the SelectKBest model and transform X in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        X_reduced : array-like of shape (n_samples, k)
            The input samples with only the selected features.
        """
        return self.fit(X, y).transform(X)

def f_classif(X, y):
    """
    Compute the ANOVA F-value for the provided sample.

    The F-value measures the linear dependency between the feature and the target. It is 
    commonly used for feature selection in classification tasks.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.

    Returns
    -------
    f_stat : array-like of shape (n_features,)
        The F-values for each feature.
    """
    num_features = X.shape[1]
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    # Calculate the mean of each feature
    mean = np.mean(X, axis=0)
    
    # Calculate the mean of each feature for each class
    mean_classes = np.zeros((num_classes, num_features))
    for i, cls in enumerate(unique_classes):
        mean_classes[i, :] = np.mean(X[y == cls], axis=0)
    
    # Calculate the between-class variance and the within-class variance
    SS_between = np.sum((mean_classes - mean) ** 2, axis=0)
    SS_within = np.sum([(X[y == cls] - mean_classes[i]) ** 2 for i, cls in enumerate(unique_classes)], axis=(0, 1))
    
    # Calculate the F-statistic
    f_stat = SS_between / (SS_within + 1e-8)
    return f_stat

