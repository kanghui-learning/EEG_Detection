import numpy as np
import xgboost as xgb
import torch
from sklearn.preprocessing import StandardScaler

class XGBoostModel:
    def __init__(self, num_classes=2, **kwargs):
        self.num_classes = num_classes
        self.model = xgb.XGBClassifier(
            objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
            num_class=num_classes if num_classes > 2 else None,
            **kwargs
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def to(self, device):
        # XGBoost doesn't use GPU in the same way as PyTorch
        return self

    def train(self):
        # XGBoost models are always in train mode
        return self

    def eval(self):
        # XGBoost models are always in eval mode
        return self

    def fit(self, X, y):
        # Reshape data for XGBoost
        X = X.reshape(X.shape[0], -1)  # Flatten the time series
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = X.reshape(X.shape[0], -1)  # Flatten the time series
        X = self.scaler.transform(X)
        return torch.from_numpy(self.model.predict_proba(X)).float()

    def __call__(self, x):
        # This method is called during forward pass in PyTorch
        return self.predict(x) 