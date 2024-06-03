import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.neural_network import MLPClassifier

# Import your custom models
from neural_networks.perceptron import Perceptron
from neural_networks.mlp import MLP

# Generate a classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test custom Perceptron
custom_perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
custom_perceptron.fit(X_train, y_train)
y_pred_custom = custom_perceptron.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Custom Perceptron accuracy: {accuracy_custom}")

# Test sklearn Perceptron
sklearn_perceptron = SklearnPerceptron()
sklearn_perceptron.fit(X_train, y_train)
y_pred_sklearn = sklearn_perceptron.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Sklearn Perceptron accuracy: {accuracy_sklearn}")

# Test custom MLP
custom_mlp = MLP(n_hidden=30, learning_rate=0.01, n_iters=1000)
custom_mlp.fit(X_train, y_train)
y_pred_custom_mlp = custom_mlp.predict(X_test)
accuracy_custom_mlp = accuracy_score(y_test, y_pred_custom_mlp)
print(f"Custom MLP accuracy: {accuracy_custom_mlp}")

# Test sklearn MLP
sklearn_mlp = MLPClassifier(hidden_layer_sizes=(30,), max_iter=1000, learning_rate_init=0.01, random_state=42)
sklearn_mlp.fit(X_train, y_train)
y_pred_sklearn_mlp = sklearn_mlp.predict(X_test)
accuracy_sklearn_mlp = accuracy_score(y_test, y_pred_sklearn_mlp)
print(f"Sklearn MLP accuracy: {accuracy_sklearn_mlp}")