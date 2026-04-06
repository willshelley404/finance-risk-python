# ties ML together
from app.ml.model import RiskModel
from app.ml.features import build_features

class RiskService:

    def __init__(self):
        self.model = RiskModel()

    def train(self, df):
        X = build_features(df)
        y = df["default"]
        self.model.train(X, y)

    def score(self, df):
        X = build_features(df)
        probs = self.model.predict(X)
        return probs