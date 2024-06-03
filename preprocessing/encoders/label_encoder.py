class LabelEncoder:
    """
    Encode labels with value between 0 and n_classes-1.

    This class transforms categorical labels to numeric labels. It can be used to preprocess labels for machine learning models.

    Attributes
    ----------
    classes_ : dict
        A dictionary mapping labels to their corresponding encoded numeric values.

    Methods
    -------
    fit(X)
        Fit label encoder by finding all unique labels in X.
    transform(X)
        Transform labels in X to numeric labels.
    fit_transform(X)
        Fit label encoder and transform labels in one step.
    """

    def __init__(self):
        self.classes_ = {}

    def fit(self, X):
        """
        Fit label encoder by finding all unique labels in X.

        Parameters
        ----------
        X : list of list of str
            The input data containing categorical labels to be encoded.

        Returns
        -------
        self : LabelEncoder
            Fitted label encoder.
        """
        unique_classes = set(val for sublist in X for val in sublist)
        self.classes_ = {label: idx for idx, label in enumerate(sorted(unique_classes))}
        return self

    def transform(self, X):
        """
        Transform labels in X to numeric labels.

        Parameters
        ----------
        X : list of list of str
            The input data containing categorical labels to be encoded.

        Returns
        -------
        list of list of int
            Transformed data with labels encoded as numeric values.
        """
        return [[self.classes_[val] for val in sublist] for sublist in X]

    def fit_transform(self, X):
        """
        Fit label encoder and transform labels in one step.

        Parameters
        ----------
        X : list of list of str
            The input data containing categorical labels to be encoded.

        Returns
        -------
        list of list of int
            Transformed data with labels encoded as numeric values.
        """
        return self.fit(X).transform(X)

