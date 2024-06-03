from preprocessing.utils.data_splitter import  train_test_split
from preprocessing.utils.missing_values import check_missing_values
import numpy as np
# Example data
X = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9], [10, np.nan, 12]])
y = np.array([0, 1, 0, 1])

# Check for missing values
missing_values = check_missing_values(X)
print("Missing values:", missing_values)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Training data:\n", X_train)
print("Testing data:\n", X_test)

