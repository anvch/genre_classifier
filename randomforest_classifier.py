from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

class RandomForestRegionClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible classifier using Random Forest
    to predict region_group of songs.
    """

    def __init__(self, n_estimators=300, max_depth=None, random_state=42, class_weight="balanced", n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.model_ = None  # Will hold the trained RandomForest

    def fit(self, X, y):
        # If embedding column exists, flatten it
        if 'embedding' in X.columns:
            X = self._flatten_embeddings(X)
        
        # Keep only numeric columns
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

        # Train the Random Forest
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs
        )
        self.model_.fit(X_numeric, y)
        return self

    def predict(self, X):
        if 'embedding' in X.columns:
            X = self._flatten_embeddings(X)
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        return self.model_.predict(X_numeric)

    def predict_proba(self, X):
        if 'embedding' in X.columns:
            X = self._flatten_embeddings(X)
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        return self.model_.predict_proba(X_numeric)

    def _flatten_embeddings(self, X):
        """Convert embedding list column into separate numeric columns"""
        emb = np.vstack(X['embedding'].values)
        emb_df = pd.DataFrame(
            emb, index=X.index,
            columns=[f'emb_{i}' for i in range(emb.shape[1])]
        )
        X_no_emb = X.drop(columns=['embedding'])
        return pd.concat([X_no_emb, emb_df], axis=1)
