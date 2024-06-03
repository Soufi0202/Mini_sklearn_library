from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from ensemble.random_forest_c import RandomForestClassifier as CustomRandomForestClassifier
from ensemble.random_forest_r import RandomForestRegressor as CustomRandomForestRegressor
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier, RandomForestRegressor as SklearnRandomForestRegressor

# Load Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split Iris dataset into training and testing sets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Custom RandomForestClassifier
custom_rf_c = CustomRandomForestClassifier(n_estimators=10, max_depth=10)
custom_rf_c.fit(X_train_iris, y_train_iris)
y_pred_custom_c = custom_rf_c.predict(X_test_iris)
accuracy_custom_c = accuracy_score(y_test_iris, y_pred_custom_c)

# Scikit-learn RandomForestClassifier
sklearn_rf_c = SklearnRandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
sklearn_rf_c.fit(X_train_iris, y_train_iris)
y_pred_sklearn_c = sklearn_rf_c.predict(X_test_iris)
accuracy_sklearn_c = accuracy_score(y_test_iris, y_pred_sklearn_c)

print(f"Iris Dataset - Custom RandomForestClassifier Accuracy: {accuracy_custom_c:.2f}")
print(f"Iris Dataset - Scikit-learn RandomForestClassifier Accuracy: {accuracy_sklearn_c:.2f}")

# Load California housing dataset
cal_housing = fetch_california_housing()
X_cal, y_cal = cal_housing.data, cal_housing.target

# Split California housing dataset into training and testing sets
X_train_cal, X_test_cal, y_train_cal, y_test_cal = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)

# Custom RandomForestRegressor
custom_rf_r = CustomRandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_impurity_decrease=0.01)
custom_rf_r.fit(X_train_cal, y_train_cal)
y_pred_custom_r = custom_rf_r.predict(X_test_cal)
mse_custom_r = mean_squared_error(y_test_cal, y_pred_custom_r)

# Scikit-learn RandomForestRegressor
sklearn_rf_r = SklearnRandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_impurity_decrease=0.01, random_state=42)
sklearn_rf_r.fit(X_train_cal, y_train_cal)
y_pred_sklearn_r = sklearn_rf_r.predict(X_test_cal)
mse_sklearn_r = mean_squared_error(y_test_cal, y_pred_sklearn_r)

print(f"California Housing Dataset - Custom RandomForestRegressor MSE: {mse_custom_r:.2f}")
print(f"California Housing Dataset - Scikit-learn RandomForestRegressor MSE: {mse_sklearn_r:.2f}")
