from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from supervised.naive_bayes import GaussianNaiveBayes

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


custom_gnb = GaussianNaiveBayes()
custom_gnb.fit(X_train, y_train)
y_pred_custom = custom_gnb.predict(X_test)

# Scikit-learn's Gaussian Naive Bayes classifier
sklearn_gnb = GaussianNB()
sklearn_gnb.fit(X_train, y_train)
y_pred_sklearn = sklearn_gnb.predict(X_test)

# Calculate accuracies
accuracy_custom = accuracy_score(y_test, y_pred_custom)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"Custom Gaussian Naive Bayes Accuracy: {accuracy_custom * 100:.2f}%")
print(f"Scikit-learn Gaussian Naive Bayes Accuracy: {accuracy_sklearn * 100:.2f}%")
