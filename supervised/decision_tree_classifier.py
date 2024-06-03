import numpy as np
from supervised.base import BaseModel

class Node:
    """
    Represents a single node in the decision tree.

    Attributes:
        feature (int): The index of the feature to split on.
        threshold (float): The threshold value for the split.
        left (Node): The left child node.
        right (Node): The right child node.
        value (any): The value to return if the node is a leaf.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Checks if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf node, otherwise False.
        """
        return self.value is not None

class DecisionTreeClassifier(BaseModel):
    """
    A decision tree classifier.

    Parameters:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_impurity_decrease (float): The minimum impurity decrease required to make a split.

    Attributes:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_impurity_decrease (float): The minimum impurity decrease required to make a split.
        root (Node): The root node of the decision tree.
    """
    def __init__(self, max_depth=100, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, X, y):
        """
        Builds the decision tree classifier from the training set (X, y).

        Parameters:
            X (array-like): The training input samples.
            y (array-like): The target values.
        """
        self.root = self._grow_tree(X, y)  

    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Parameters:
            X (array-like): The input samples.

        Returns:
            array: The predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the decision tree.

        Parameters:
            X (array-like): The input samples.
            y (array-like): The target values.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the grown tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, n_features, replace=False)

        best_feat, best_thresh, best_gain = self._best_criteria(X, y, feat_idxs)
        if best_feat is None or best_gain < self.min_impurity_decrease:
            return Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Finds the best feature and threshold for splitting the data.

        Parameters:
            X (array-like): The input samples.
            y (array-like): The target values.
            feat_idxs (array-like): The indices of the features to consider.

        Returns:
            tuple: The index of the best feature, the best threshold, and the best information gain.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh, best_gain

    def _information_gain(self, y, X_column, split_thresh):
        """
        Computes the information gain of a split.

        Parameters:
            y (array-like): The target values.
            X_column (array-like): A column of feature values.
            split_thresh (float): The threshold to split on.

        Returns:
            float: The information gain of the split.
        """
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n, n_left, n_right = len(y), len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        """
        Splits the data into left and right branches.

        Parameters:
            X_column (array-like): A column of feature values.
            split_thresh (float): The threshold to split on.

        Returns:
            tuple: The indices of the left and right branches.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Computes the entropy of the target values.

        Parameters:
            y (array-like): The target values.

        Returns:
            float: The entropy of the target values.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Finds the most common label in the target values.

        Parameters:
            y (array-like): The target values.

        Returns:
            int: The most common label.
        """
        return np.bincount(y).argmax()

    def _traverse_tree(self, x, node):
        """
        Traverses the tree to make a prediction for a single sample.

        Parameters:
            x (array-like): A single input sample.
            node (Node): The current node in the tree.

        Returns:
            int: The predicted class label.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
