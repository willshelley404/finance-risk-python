from fastapi import FastAPI
from pydantic import BaseModel
from app.data.synthetic import generate_data
from app.services.risk_service import RiskService
import pandas as pd

app = FastAPI(title="Finance Risk Assessment API")

service = RiskService()
training_data = generate_data(200)
service.train(training_data)

class UserProfile(BaseModel):
    age: int
    monthly_income: float
    monthly_expenses: float
    total_debt: float
    credit_score: int
    savings: float
    credit_card_balance: float
    credit_limit: float
    late_payments_12m: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/assess")
def assess_user(profile: UserProfile):
    user_df = pd.DataFrame([profile.dict()])
    assessment = service.assess_risk(user_df)
    return assessment

@app.post("/explain")
def explain_risk(profile: UserProfile):
    user_df = pd.DataFrame([profile.dict()])
    assessment = service.assess_risk(user_df)
    explanation = service.get_explanation(user_df, assessment)
    return {
        "assessment": assessment,
        "explanation": explanation
    }
