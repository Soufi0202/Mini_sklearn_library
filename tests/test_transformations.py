from preprocessing.transformations.polynomial_features import PolynomialFeatures
import numpy as np

# Data for testing
data = np.array([[1, 2], [3, 4]])



# Test Polynomial Features
poly_features = PolynomialFeatures(degree=3)
poly_data = poly_features.fit_transform(data)
print("Polynomial Features Transformed Data:\n", poly_data)