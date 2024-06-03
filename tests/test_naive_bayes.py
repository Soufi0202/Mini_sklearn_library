# Test the GaussianNaiveBayes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from supervised.naive_bayes import GaussianNaiveBayes


iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gaussian Naive Bayes classifier
nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)

# Predict on the test set
y_pred = nb.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

