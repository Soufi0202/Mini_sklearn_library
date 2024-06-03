from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from ensemble.bagging import BaggingClassifier
from ensemble.boosting import AdaBoostClassifier
from ensemble.boosting import GradientBoostingClassifier
from model_selection.metrics import accuracy_score



data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Bagging classifier
bagging_clf = BaggingClassifier(base_model=DecisionTreeClassifier, n_estimators=10, max_samples=1.0)
bagging_clf.fit(X_train, y_train)

# Make predictions
y_pred = bagging_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

y = np.where(y == 2, 1, y)  # Convert to binary classification (0 and 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the AdaBoost classifier
ada_clf = AdaBoostClassifier(base_model=DecisionTreeClassifier, n_estimators=50, learning_rate=1.0)
ada_clf.fit(X_train, y_train)

# Make predictions
y_pred = ada_clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Load data
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Gradient Boosting Classifier
gbc = GradientBoostingClassifier(base_model=DecisionTreeRegressor, n_estimators=100, learning_rate=0.1)
gbc.fit(X_train, y_train)

# Make predictions
y_pred = gbc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on Wine dataset: {accuracy * 100:.2f}%")