import pandas as pd
import numpy as np
from app.ml.model import RiskModel
from app.ml.features import FeatureEngineer
from app.ml.explain import RiskExplainer
from app.agent.agent import FinanceAgent

class RiskService:
    def __init__(self):
        self.model = RiskModel()
        self.explainer = None
        self.agent = FinanceAgent()
        self.feature_engineer = FeatureEngineer()
    
    def train(self, df_train):
        X = self.feature_engineer.create_features(df_train)
        y = df_train['default']
        self.model.train(X, y)
        self.explainer = RiskExplainer(self.model)
        self.explainer.fit(X)
    
    def assess_risk(self, user_data):
        X = self.feature_engineer.create_features(user_data)
        risk_score = self.model.predict(X)[0]
        explanation = self.explainer.explain_prediction(X)
        return {
            'risk_score': risk_score,
            'explanation': explanation,
            'high_risk': risk_score > 0.5
        }
    
    def get_explanation(self, user_data, risk_assessment):
        return self.agent.get_recommendations(
            risk_assessment['risk_score'],
            risk_assessment['explanation']['feature_importance']
        )
