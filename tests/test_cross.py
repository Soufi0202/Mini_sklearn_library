from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
from model_selection.cross_validation import cross_validate


# 1. Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 2. Define a model
model = SVC()


# 3. Perform cross-validation using scikit-learn's cross-validation functions
scikit_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# 4. Evaluate the results
print("Mean score from scikit-learn's cross_val_score:", np.mean(scikit_scores))
print("Standard deviation of scores from scikit-learn's cross_val_score:", np.std(scikit_scores))

# Now, let's test our custom cross_validate function
mean_score, std_score = cross_validate(model, X, y)

print("\nMean score from custom cross_validate:", mean_score)
print("Standard deviation of scores from custom cross_validate:", std_score)
