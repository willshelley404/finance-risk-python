# scenario analysis
def simulate_utilization(df, new_util):
    df = df.copy()
    df["credit_utilization"] = new_util
    return df