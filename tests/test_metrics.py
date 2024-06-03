from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from model_selection.metrics import accuracy_score , precision_score ,recall_score,f1_score,confusion_matrix,plot_confusion_matrix

# Define your custom functions here

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics using our custom functions
acc_custom = accuracy_score(y_test, y_pred)
prec_custom = precision_score(y_test, y_pred)
recall_custom = recall_score(y_test, y_pred)
f1_custom = f1_score(y_test, y_pred)

# Calculate evaluation metrics using scikit-learn's implementations
acc_sklearn = sklearn_accuracy_score(y_test, y_pred)
prec_sklearn = sklearn_precision_score(y_test, y_pred)
recall_sklearn = sklearn_recall_score(y_test, y_pred)
f1_sklearn = sklearn_f1_score(y_test, y_pred)

# Compare the results
print("Accuracy (Custom):", acc_custom)
print("Accuracy (Scikit-learn):", acc_sklearn)

print("Precision (Custom):", prec_custom)
print("Precision (Scikit-learn):", prec_sklearn)

print("Recall (Custom):", recall_custom)
print("Recall (Scikit-learn):", recall_sklearn)

print("F1 Score (Custom):", f1_custom)
print("F1 Score (Scikit-learn):", f1_sklearn)

# Plot confusion matrix using our custom function
cm_custom = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm_custom, labels=['Class 0', 'Class 1'], title='Confusion Matrix (Custom)')

# Plot confusion matrix using scikit-learn's implementation
cm_sklearn = sklearn_confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm_sklearn, labels=['Class 0', 'Class 1'], title='Confusion Matrix (Scikit-learn)')

plt.show()
