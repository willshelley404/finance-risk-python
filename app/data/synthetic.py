import numpy as np
import pandas as pd

def generate_data(n=1000):
    np.random.seed(42)

    age = np.random.randint(18, 70, n)
    income = np.random.normal(5000, 1500, n).clip(1000)
    expenses = income * np.random.uniform(0.5, 1.2, n)
    debt = np.random.normal(30000, 20000, n).clip(0)
    credit_score = np.random.normal(680, 80, n).clip(300, 850)
    savings = np.random.normal(10000, 8000, n).clip(0)
    cc_balance = np.random.normal(5000, 3000, n).clip(0)
    credit_limit = cc_balance + np.random.normal(10000, 5000, n).clip(1000)
    late_payments = np.random.poisson(1.5, n)

    # Derived features (THIS IS KEY)
    utilization = cc_balance / credit_limit
    dti = debt / income

    # Risk function (logistic style)
    risk_score = (
        2.5 * utilization +
        1.5 * dti +
        -0.003 * credit_score +
        0.8 * late_payments +
        -0.00005 * savings +
        np.random.normal(0, 0.5, n)  # noise
    )

    prob_default = 1 / (1 + np.exp(-risk_score))
    default = (prob_default > 0.5).astype(int)

    return pd.DataFrame({
        "age": age,
        "monthly_income": income,
        "monthly_expenses": expenses,
        "total_debt": debt,
        "credit_score": credit_score,
        "savings": savings,
        "credit_card_balance": cc_balance,
        "credit_limit": credit_limit,
        "late_payments_12m": late_payments,
        "default": default
    })