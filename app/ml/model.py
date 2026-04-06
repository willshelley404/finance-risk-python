# training + prediction
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class RiskModel:

    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05
        )

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.model.fit(X_train, y_train)
        print("Model trained")

    def predict(self, X):
        prob = self.model.predict_proba(X)[:, 1]
        return prob