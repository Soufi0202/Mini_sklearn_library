from preprocessing.encoders.label_encoder import LabelEncoder
from preprocessing.encoders.onehot_encoder import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits



iris_data = load_iris()
digits_data = load_digits()


# Label Encoder Test
label_encoder = LabelEncoder()
label_encoded_data = label_encoder.fit_transform(iris_data)
print("Label Encoded Data:", label_encoded_data)

# One-Hot Encoder Test
onehot_encoder = OneHotEncoder()
onehot_encoded_data = onehot_encoder.fit_transform(iris_data)
print("One-Hot Encoded Data:\n", onehot_encoded_data)