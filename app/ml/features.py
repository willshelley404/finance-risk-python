# feature engineering
def build_features(df):
    df = df.copy()

    df["dti"] = df["debt"] / df["income"]
    df["burn_rate"] = df["expenses"] / df["income"]
    df["savings_ratio"] = df["savings"] / df["income"]

    return df[[
        "income", "expenses", "savings",
        "debt", "credit_utilization",
        "missed_payments", "dti",
        "burn_rate", "savings_ratio"
    ]]