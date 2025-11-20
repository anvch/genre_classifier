from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

class XGBoostGenreClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 subsample=0.8, colsample_bytree=0.8, random_state=42, 
                 tree_method='hist', verbosity=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.tree_method = tree_method
        self.verbosity = verbosity
        self.model_ = None
        
        

    def fit(self, X, y):
        # If embedding column exists, flatten it
        if 'embedding' in X.columns:
            X = self._flatten_embeddings(X)
        
        # Keep only numeric columns
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

        # y is already encoded as integers from eval.py
        y_encoded = y
        
        # Store unique classes and count them
        self.classes_ = np.unique(y_encoded)
        num_classes = len(self.classes_)

        # Train XGBoost
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            tree_method=self.tree_method,
            verbosity=self.verbosity,
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss'
        )
        self.model_.fit(X_numeric, y_encoded)
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
        proba = self.model_.predict_proba(X_numeric)
        return proba / proba.sum(axis=1, keepdims=True)

    def _flatten_embeddings(self, X):
        """Convert embedding list column into separate numeric columns"""
        emb = np.vstack(X['embedding'].values)
        emb_df = pd.DataFrame(
            emb, index=X.index,
            columns=[f'emb_{i}' for i in range(emb.shape[1])]
        )
        X_no_emb = X.drop(columns=['embedding'])
        return pd.concat([X_no_emb, emb_df], axis=1)
