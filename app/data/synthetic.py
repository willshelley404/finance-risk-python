# fake data generator
import pandas as pd
import numpy as np

def generate_data(n=1000):
    np.random.seed(42)

    df = pd.DataFrame({
        "income": np.random.normal(5000, 1500, n),
        "expenses": np.random.normal(4000, 1200, n),
        "savings": np.random.normal(10000, 5000, n),
        "debt": np.random.normal(15000, 8000, n),
        "credit_utilization": np.random.uniform(0.1, 0.9, n),
        "missed_payments": np.random.randint(0, 5, n)
    })

    df["dti"] = df["debt"] / df["income"]
    df["burn_rate"] = df["expenses"] / df["income"]

    # target
    df["default"] = (
        (df["credit_utilization"] > 0.6) |
        (df["missed_payments"] > 2) |
        (df["dti"] > 0.5)
    ).astype(int)

    return df