# agent tools
def format_features(row):
    return {
        "credit_utilization": row["credit_utilization"],
        "dti": row["dti"],
        "savings": row["savings"]
    }

def risk_explanation_tool(features):
    explanations = []

    if features["credit_utilization"] > 0.7:
        explanations.append("High credit utilization increases default risk.")

    if features["dti"] > 0.5:
        explanations.append("High debt-to-income ratio indicates financial stress.")

    if features["savings"] < 2000:
        explanations.append("Low savings reduces financial buffer.")

    return explanations