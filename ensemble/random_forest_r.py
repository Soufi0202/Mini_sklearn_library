import numpy as np
from supervised.decisiontree_regressor import DecisionTreeRegressor

class RandomForestRegressor:
    """
    RandomForestRegressor is an ensemble learning method that constructs a multitude of decision trees 
    during training and outputs the average prediction of the individual trees.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=10
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until 
        all leaves contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.

    Attributes
    ----------
    trees : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y)
        Build a forest of trees from the training set (X, y).
    predict(X)
        Predict regression target for X.
    _bootstrap_sample(X, y)
        Generate a bootstrap sample from the dataset (X, y).
    """

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2, min_impurity_decrease=0.0, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (real numbers).
        """
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted value for an input sample is computed as the mean predicted values of the individual trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted values.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)

    def _bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample from the dataset (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        X_sample : array-like of shape (n_samples, n_features)
            The bootstrap sample of input samples.
        y_sample : array-like of shape (n_samples,)
            The bootstrap sample of target values.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
