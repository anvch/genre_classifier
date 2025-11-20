from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class SVMGenreClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42, class_weight="balanced", probability=True):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.class_weight = class_weight
        self.probability = probability
        self.model_ = None
        self.scaler_ = None

    def fit(self, X, y):
        # If embedding column exists, flatten it
        if 'embedding' in X.columns:
            X = self._flatten_embeddings(X)
        
        # Keep only numeric columns
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)

        # SVMs are sensitive to feature scaling, so standardize the features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_numeric)

        # Train the SVM
        self.model_ = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state,
            class_weight=self.class_weight,
            probability=self.probability
        )
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X):
        if 'embedding' in X.columns:
            X = self._flatten_embeddings(X)
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler_.transform(X_numeric)
        return self.model_.predict(X_scaled)

    def predict_proba(self, X):
        if 'embedding' in X.columns:
            X = self._flatten_embeddings(X)
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler_.transform(X_numeric)
        return self.model_.predict_proba(X_scaled)

    def _flatten_embeddings(self, X):
        """Convert embedding list column into separate numeric columns"""
        emb = np.vstack(X['embedding'].values)
        emb_df = pd.DataFrame(
            emb, index=X.index,
            columns=[f'emb_{i}' for i in range(emb.shape[1])]
        )
        X_no_emb = X.drop(columns=['embedding'])
        return pd.concat([X_no_emb, emb_df], axis=1)

    @property
    def classes_(self):
        """Return the classes attribute from the underlying SVC model"""
        return self.model_.classes_