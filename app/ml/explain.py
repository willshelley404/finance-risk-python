import shap
import pandas as pd
import numpy as np

class RiskExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = None
        self.shap_values = None
    
    def fit(self, X_train):
        self.explainer = shap.TreeExplainer(self.model.model)
        self.shap_values = self.explainer.shap_values(self.model.scaler.transform(X_train))
    
    def explain_prediction(self, X_sample):
        X_scaled = self.model.scaler.transform(X_sample)
        shap_vals = self.explainer.shap_values(X_scaled)
        
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        
        explanation = {
            'prediction': self.model.predict(X_sample)[0],
            'feature_importance': dict(zip(self.model.feature_names, shap_vals[0])),
            'features': X_sample.to_dict('records')[0]
        }
        return explanation
