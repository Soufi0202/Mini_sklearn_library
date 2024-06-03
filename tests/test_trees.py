import numpy as np
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from supervised.decisiontree_regressor import DecisionTreeRegressor
from supervised.decision_tree_classifier import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

# California Housing dataset - Regression
cal_housing = fetch_california_housing()
X, y = cal_housing.data, cal_housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Decision Tree Regressor
regressor = DecisionTreeRegressor(max_depth=10, min_samples_split=2, min_impurity_decrease=0.01)
regressor.fit(X_train, y_train)
y_pred_custom = regressor.predict(X_test)
mse_custom = mean_squared_error(y_test, y_pred_custom)
print(f"Mean Squared Error (Custom Decision Tree): {mse_custom:.2f}")

# Sklearn Decision Tree Regressor
sklearn_regressor = SklearnDecisionTreeRegressor(max_depth=10, min_samples_split=2, min_impurity_decrease=0.01)
sklearn_regressor.fit(X_train, y_train)
y_pred_sklearn = sklearn_regressor.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print(f"Mean Squared Error (Sklearn Decision Tree): {mse_sklearn:.2f}")


# Wine dataset - Classification
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_impurity_decrease=0.01)
clf.fit(X_train, y_train)
y_pred_custom = clf.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Accuracy on Wine dataset (Custom Decision Tree): {accuracy_custom * 100:.2f}%")

# Sklearn Decision Tree Classifier
sklearn_clf = SklearnDecisionTreeClassifier(max_depth=10, min_samples_split=2, min_impurity_decrease=0.01)
sklearn_clf.fit(X_train, y_train)
y_pred_sklearn = sklearn_clf.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Accuracy on Wine dataset (Sklearn Decision Tree): {accuracy_sklearn * 100:.2f}%")
