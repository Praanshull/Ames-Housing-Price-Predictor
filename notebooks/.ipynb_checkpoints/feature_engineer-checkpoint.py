# feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # whatever transformations you used in your notebook
        X = X.copy()
        X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
        return X
