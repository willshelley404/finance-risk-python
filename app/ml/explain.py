# SHAP logic
import shap

class Explainer:

    def __init__(self, model):
        self.explainer = shap.Explainer(model)

    def explain(self, X):
        shap_values = self.explainer(X)
        return shap_values