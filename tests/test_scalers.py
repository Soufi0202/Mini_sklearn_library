import numpy as np
from preprocessing.scalers.standard_scaler import StandardScaler
from preprocessing.scalers.maxabs_scaler import MaxAbsScaler
from preprocessing.scalers.minmax_scaler import MinMaxScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler

# Sample data
X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])

# Custom Standard Scaler
custom_std_scaler = StandardScaler()
X_std_custom = custom_std_scaler.fit_transform(X)

# Custom Min-Max Scaler
custom_minmax_scaler = MinMaxScaler()
X_minmax_custom = custom_minmax_scaler.fit_transform(X)

# Custom MaxAbs Scaler
custom_maxabs_scaler = MaxAbsScaler()
X_maxabs_custom = custom_maxabs_scaler.fit_transform(X)

# Scikit-Learn Standard Scaler
sklearn_std_scaler = SklearnStandardScaler()
X_std_sklearn = sklearn_std_scaler.fit_transform(X)

# Scikit-Learn Min-Max Scaler
sklearn_minmax_scaler = SklearnMinMaxScaler()
X_minmax_sklearn = sklearn_minmax_scaler.fit_transform(X)

# Scikit-Learn MaxAbs Scaler
sklearn_maxabs_scaler = SklearnMaxAbsScaler()
X_maxabs_sklearn = sklearn_maxabs_scaler.fit_transform(X)

# Compare the results
print("Custom Standard Scaler:\n", X_std_custom)
print("Scikit-Learn Standard Scaler:\n", X_std_sklearn)

print("Custom Min-Max Scaler:\n", X_minmax_custom)
print("Scikit-Learn Min-Max Scaler:\n", X_minmax_sklearn)

print("Custom MaxAbs Scaler:\n", X_maxabs_custom)
print("Scikit-Learn MaxAbs Scaler:\n", X_maxabs_sklearn)
