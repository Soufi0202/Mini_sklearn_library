import numpy as np

def check_missing_values(X):
    """
    Check for missing values in the dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    missing_values : dict
        Dictionary with feature indices as keys and counts of missing values as values.
    """
    if len(X) == 0:
        raise ValueError("Input array X is empty")
    missing_values = {}
    for i in range(X.shape[1]):
        missing_count = np.sum(np.isnan(X[:, i]))
        if missing_count > 0:
            missing_values[i] = missing_count
    return missing_values
