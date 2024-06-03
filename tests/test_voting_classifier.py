import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from supervised.linear_models import LinearRegression
from supervised.decision_tree_classifier import DecisionTreeClassifier
from supervised.knn import KNN
from ensemble.voting_classifier import VotingClassifier

def test_voting_classifier():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = [DecisionTreeClassifier(max_depth=10), KNN(k=3), LinearRegression()]
    voting_clf = VotingClassifier(models)
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_voting_classifier()
