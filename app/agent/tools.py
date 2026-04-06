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

def shap_to_text(feature_names, shap_values, feature_values):

    explanations = []

    for name, val, feat in zip(feature_names, shap_values, feature_values):

        if abs(val) < 0.05:
            continue

        direction = "increases" if val > 0 else "reduces"

        explanations.append(
            f"{name} ({feat:.2f}) {direction} risk"
        )

    return explanations