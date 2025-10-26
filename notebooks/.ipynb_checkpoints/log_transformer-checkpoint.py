from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # If X is a NumPy array
        if isinstance(X, np.ndarray):
            return np.log1p(X)
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = np.log1p(X_copy[col])
        return X_copy
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)