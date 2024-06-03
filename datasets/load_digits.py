import pandas as pd
import numpy as np

def load_digits():
    """
    Load and return the digits dataset (classification).

    This function creates a synthetic digits dataset similar to the one provided by sklearn,
    which is commonly used for classification tasks. Each feature represents the intensity
    of a pixel in an 8x8 image.

    Returns
    -------
    data : DataFrame
        The data frame containing the digits dataset with columns:
        - 'pixel_0'
        - 'pixel_1'
        - ...
        - 'pixel_63'
        - 'target'
    """
    # Generate synthetic digits dataset
    from sklearn.datasets import load_digits as sklearn_load_digits
    digits = sklearn_load_digits()

    # Create a DataFrame
    df = pd.DataFrame(data=digits.data, columns=[f'pixel_{i}' for i in range(digits.data.shape[1])])
    df['target'] = digits.target
    return df


