from preprocessing.normalizers.l1_normalizer import L1Normalizer
from sklearn.preprocessing import Normalizer as SklearnNormalizer
import numpy as np

# Sample data
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Custom L1 Normalizer
custom_l1_normalizer = L1Normalizer()
X_l1_custom = custom_l1_normalizer.fit_transform(X)


# Scikit-Learn Normalizer
sklearn_normalizer = SklearnNormalizer(norm='l1')  # Choose the appropriate norm
X_sklearn = sklearn_normalizer.fit_transform(X)

# Compare the results
print("Custom L1 Normalized data:\n", X_l1_custom)
print("Scikit-Learn L1 Normalized data:\n", X_sklearn)


