import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

class RiskModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X, y):
        """Train the model on historical data"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.feature_names = X.columns.tolist()
        
    def predict(self, X):
        """Predict probability of default"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, path: str):
        """Save model to disk"""
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_names}, f)
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['features']