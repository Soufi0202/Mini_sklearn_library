class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError("Each model must re-implement this method.")

    def predict(self, X):
        raise NotImplementedError("Each model must re-implement this method.")
