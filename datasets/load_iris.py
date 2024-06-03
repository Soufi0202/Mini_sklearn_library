import pandas as pd
from sklearn.datasets import load_iris as sklearn_load_iris

def load_iris():
    """
    Load and return the iris dataset (classification).
    
    The iris dataset is a classic dataset for classification and machine learning. 
    It contains 150 observations of iris flowers, with four features (sepal length, sepal width,
    petal length, and petal width) and a target label indicating the species of iris 
    (Setosa, Versicolour, or Virginica).
    
    Returns
    -------
    data : DataFrame
        The data frame containing the iris dataset with columns:
        - 'sepal length (cm)'
        - 'sepal width (cm)'
        - 'petal length (cm)'
        - 'petal width (cm)'
        - 'target'
    """
    iris = sklearn_load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df
