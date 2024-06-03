import numpy as np

class SimpleImputer:
    """
    Impute missing values using simple strategies.

    This imputer provides basic strategies for imputing missing values. The supported 
    strategies are 'mean', 'median', 'most_frequent', and 'constant'.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy.
        - If 'mean', replace missing values using the mean along each column.
        - If 'median', replace missing values using the median along each column.
        - If 'most_frequent', replace missing values using the most frequent value along each column.
        - If 'constant', replace missing values with fill_value.
    fill_value : any, default=None
        When strategy is 'constant', this is the value used to replace missing values. If None, it raises an error.

    Attributes
    ----------
    statistics_ : array-like of shape (n_features,)
        The statistics for each feature used for imputation.

    Methods
    -------
    fit(X, y=None)
        Compute the statistics for each feature.
    transform(X)
        Impute missing values in X using the fitted statistics.
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X, y=None):
        """
        Compute the statistics for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        self : object
            Fitted SimpleImputer model.
        """
        if self.strategy not in ['mean', 'median', 'most_frequent', 'constant']:
            raise ValueError("Invalid strategy: {}".format(self.strategy))
        
        if self.strategy == 'constant' and self.fill_value is None:
            raise ValueError("fill_value cannot be None when strategy is 'constant'")
        
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = [np.bincount(column[~np.isnan(column)].astype(int)).argmax() for column in X.T]
        elif self.strategy == 'constant':
            self.statistics_ = np.full((X.shape[1],), self.fill_value, dtype=X.dtype)
        
        return self

    def transform(self, X):
        """
        Impute missing values in X using the fitted statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples with missing values.

        Returns
        -------
        X_out : array-like of shape (n_samples, n_features)
            The input samples with imputed values.
        """
        mask = np.isnan(X)
        X_out = np.array(X, copy=True)
        for i, stat in enumerate(self.statistics_):
            X_out[mask[:, i], i] = stat
        return X_out

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples with missing values.
        y : None
            Ignored. This parameter exists for compatibility with sklearn's fit method.

        Returns
        -------
        X_out : array-like of shape (n_samples, n_features)
            The input samples with imputed values.
        """
        return self.fit(X, y).transform(X)

