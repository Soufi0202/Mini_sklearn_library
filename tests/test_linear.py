from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from supervised.linear_models import LinearRegression
from supervised.linear_models import LogisticRegression

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = load_diabetes()

# Split the dataset into training and testing sets
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Linear Regression
custom_linear_regression = LinearRegression()
custom_linear_regression.fit(X_diabetes_train, y_diabetes_train)
y_diabetes_pred_custom = custom_linear_regression.predict(X_diabetes_test)

sklearn_linear_regression = SklearnLinearRegression()
sklearn_linear_regression.fit(X_diabetes_train, y_diabetes_train)
y_diabetes_pred_sklearn = sklearn_linear_regression.predict(X_diabetes_test)

mse_custom = mean_squared_error(y_diabetes_test, y_diabetes_pred_custom)
mse_sklearn = mean_squared_error(y_diabetes_test, y_diabetes_pred_sklearn)

print("Mean Squared Error (Custom Linear Regression):", mse_custom)
print("Mean Squared Error (Scikit-learn Linear Regression):", mse_sklearn)


# Load the diabetes dataset
diabetes = load_diabetes()

# Bin the target variable to perform binary classification
y_diabetes_binary = (diabetes.target > diabetes.target.mean()).astype(int)

# Split the dataset into training and testing sets
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(diabetes.data, y_diabetes_binary, test_size=0.2, random_state=42)

# Logistic Regression
custom_logistic_regression = LogisticRegression()
custom_logistic_regression.fit(X_diabetes_train, y_diabetes_train)
y_diabetes_pred_custom = custom_logistic_regression.predict(X_diabetes_test)

sklearn_logistic_regression = SklearnLogisticRegression()
sklearn_logistic_regression.fit(X_diabetes_train, y_diabetes_train)
y_diabetes_pred_sklearn = sklearn_logistic_regression.predict(X_diabetes_test)

accuracy_custom = accuracy_score(y_diabetes_test, y_diabetes_pred_custom)
accuracy_sklearn = accuracy_score(y_diabetes_test, y_diabetes_pred_sklearn)

print("Accuracy (Custom Logistic Regression):", accuracy_custom)
print("Accuracy (Scikit-learn Logistic Regression):", accuracy_sklearn)

