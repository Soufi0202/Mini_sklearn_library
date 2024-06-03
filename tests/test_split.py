from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import numpy as np
from model_selection.split import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Spliting the dataset using our custom train_test_split function
X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the dataset using scikit-learn's train_test_split function
X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)

# Compare the results
# Check if the shapes of training and testing sets are the same
assert X_train_custom.shape == X_train_sklearn.shape
assert X_test_custom.shape == X_test_sklearn.shape
assert y_train_custom.shape == y_train_sklearn.shape
assert y_test_custom.shape == y_test_sklearn.shape

# Check if the data points are the same
assert np.array_equal(X_train_custom, X_train_sklearn)
assert np.array_equal(X_test_custom, X_test_sklearn)
assert np.array_equal(y_train_custom, y_train_sklearn)
assert np.array_equal(y_test_custom, y_test_sklearn)

print("Custom train_test_split and scikit-learn's train_test_split produce consistent results.")
