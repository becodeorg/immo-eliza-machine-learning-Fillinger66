from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TopKOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.top_categories_ = {}
        self.columns_ = []

    def fit(self, X, y=None):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        else:
            X = X.copy()
        # Get columns list
        self.columns_ = X.columns
        # Get index of the K top values
        for col in self.columns_:
            top = X[col].value_counts().nlargest(self.top_k).index
            self.top_categories_[col] = top
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        else:
            X = X.copy()

        X_encoded = pd.DataFrame(index=X.index)
        
        for col in self.columns_:
            top = self.top_categories_[col]
            for category in top:
                X_encoded[f"{col}_{category}"] = (X[col] == category).astype(int)

        return X_encoded
