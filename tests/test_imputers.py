from preprocessing.imputers.simple_imputer import SimpleImputer
from preprocessing.imputers.knn_imputer import KNNImputer
from sklearn.impute import SimpleImputer as SklearnSimpleImputer
from sklearn.impute import KNNImputer as SklearnKNNImputer
import numpy as np

# Sample data with missing values
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])

# Custom Simple Imputer
custom_simple_imputer_mean = SimpleImputer(strategy='mean')
custom_simple_imputer_mean_data = custom_simple_imputer_mean.fit_transform(data)

# Scikit-Learn Simple Imputer
sklearn_simple_imputer_mean = SklearnSimpleImputer(strategy='mean')
sklearn_simple_imputer_mean_data = sklearn_simple_imputer_mean.fit_transform(data)

# Compare Simple Imputer results
print("Custom Simple Imputer with Mean Strategy:\n", custom_simple_imputer_mean_data)
print("Scikit-Learn Simple Imputer with Mean Strategy:\n", sklearn_simple_imputer_mean_data)

# Custom Simple Imputer with constant strategy
custom_simple_imputer_constant = SimpleImputer(strategy='constant', fill_value=999)
custom_simple_imputer_constant_data = custom_simple_imputer_constant.fit_transform(data)

# Scikit-Learn Simple Imputer with constant strategy
sklearn_simple_imputer_constant = SklearnSimpleImputer(strategy='constant', fill_value=999)
sklearn_simple_imputer_constant_data = sklearn_simple_imputer_constant.fit_transform(data)

# Compare Simple Imputer results with constant strategy
print("Custom Simple Imputer with Constant Strategy:\n", custom_simple_imputer_constant_data)
print("Scikit-Learn Simple Imputer with Constant Strategy:\n", sklearn_simple_imputer_constant_data)


# Sample data with missing values
data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9], [np.nan, 6, 5], [8, 8, 8]])

# Custom KNN Imputer
custom_knn_imputer = KNNImputer(n_neighbors=2)
custom_knn_imputer_data = custom_knn_imputer.fit_transform(data)

# Scikit-Learn KNN Imputer
sklearn_knn_imputer = SklearnKNNImputer(n_neighbors=2)
sklearn_knn_imputer_data = sklearn_knn_imputer.fit_transform(data)

# Compare KNN Imputer results
print("Custom KNN Imputer:\n", custom_knn_imputer_data)
print("Scikit-Learn KNN Imputer:\n", sklearn_knn_imputer_data)
