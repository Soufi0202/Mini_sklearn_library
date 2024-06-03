from datasets.load_iris import load_iris
from datasets.load_digits import load_digits

iris_data = load_iris()
print(iris_data.head())

digits_data = load_digits()
print(digits_data.head())