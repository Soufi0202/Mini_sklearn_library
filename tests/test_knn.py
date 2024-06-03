from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your KNN implementation
from supervised.knn import KNN
custom_knn = KNN(k=3)
custom_knn.fit(X_train, y_train)
y_pred_custom = custom_knn.predict(X_test)

# Scikit-learn's KNN classifier
sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train, y_train)
y_pred_sklearn = sklearn_knn.predict(X_test)

# Calculate accuracies
accuracy_custom = accuracy_score(y_test, y_pred_custom)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"Custom KNN Accuracy: {accuracy_custom * 100:.2f}%")
print(f"Scikit-learn KNN Accuracy: {accuracy_sklearn * 100:.2f}%")
