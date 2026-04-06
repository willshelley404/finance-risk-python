# Streamlit app frontend
from fastapi import FastAPI
import pandas as pd

from app.data.synthetic import generate_data
from app.services.risk_service import RiskService

app = FastAPI()

service = RiskService()

# train on startup
df = generate_data(1000)
service.train(df)

@app.get("/score")
def score():

    sample = df.sample(1)
    prob = service.score(sample)[0]

    return {
        "risk_score": float(prob),
        "input": sample.to_dict(orient="records")[0]
    }