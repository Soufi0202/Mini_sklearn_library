from sklearn.datasets import load_iris
from preprocessing.feature_selection.selectkbest import  SelectKBest, f_classif
from preprocessing.feature_selection.variance import VarianceThreshold
from preprocessing.feature_selection.rfe import RFE, SimpleTreeEstimator
from sklearn.feature_selection import VarianceThreshold as SklearnVarianceThreshold
from sklearn.feature_selection import SelectKBest as SklearnSelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier



# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Variance Threshold
vt = VarianceThreshold(threshold=0.2)
X_vt = vt.fit_transform(X)
print("Variance Threshold:\n", X_vt)

# SelectKBest
skb = SelectKBest(score_func=f_classif, k=2)
X_skb = skb.fit_transform(X, y)
print("SelectKBest:\n", X_skb)

# Recursive Feature Elimination
estimator = SimpleTreeEstimator()
rfe = RFE(estimator=estimator, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
print("RFE:\n", X_rfe)


# Variance Threshold
sklearn_vt = SklearnVarianceThreshold(threshold=0.2)
X_vt_sklearn = sklearn_vt.fit_transform(X)
print("Variance Threshold (Scikit-learn):\n", X_vt_sklearn)

# SelectKBest
sklearn_skb = SklearnSelectKBest(score_func=f_classif, k=2)
X_skb_sklearn = sklearn_skb.fit_transform(X, y)
print("SelectKBest (Scikit-learn):\n", X_skb_sklearn)

# Recursive Feature Elimination
estimator = DecisionTreeClassifier()
rfe_sklearn = RFE(estimator=estimator, n_features_to_select=2)
X_rfe_sklearn = rfe_sklearn.fit_transform(X, y)
print("RFE (Scikit-learn):\n", X_rfe_sklearn)