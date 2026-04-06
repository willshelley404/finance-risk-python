import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from app.data.synthetic import generate_data
from app.services.risk_service import RiskService

st.set_page_config(page_title="Finance Risk Predictor", layout="wide")
st.title("Personal Finance Risk Predictor")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Select Mode", ["Assess Risk", "Train Model"])

service = RiskService()
training_data = generate_data(200)
service.train(training_data)

if mode == "Assess Risk":
    st.subheader("Individual Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)
        monthly_expenses = st.number_input("Monthly Expenses", min_value=500, value=3000)
        total_debt = st.number_input("Total Debt", min_value=0, value=50000)
    
    with col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        savings = st.number_input("Savings", min_value=0, value=10000)
        credit_card_balance = st.number_input("Credit Card Balance", min_value=0, value=5000)
        credit_limit = st.number_input("Credit Limit", min_value=1000, value=20000)
        late_payments_12m = st.number_input("Late Payments (12m)", min_value=0, max_value=12, value=0)
    
    if st.button("Assess Risk"):
        user_data = pd.DataFrame([{
            'age': age,
            'monthly_income': monthly_income,
            'monthly_expenses': monthly_expenses,
            'total_debt': total_debt,
            'credit_score': credit_score,
            'savings': savings,
            'credit_card_balance': credit_card_balance,
            'credit_limit': credit_limit,
            'late_payments_12m': late_payments_12m
        }])
        
        assessment = service.assess_risk(user_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Default Risk", f"{assessment['risk_score']:.1%}")
        with col2:
            status = "⚠️ HIGH RISK" if assessment['high_risk'] else "✅ LOW RISK"
            st.metric("Status", status)
        
        st.subheader("Risk Factors")
        factors = assessment['explanation']['feature_importance']
        factor_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Impact'])
        st.bar_chart(factor_df.set_index('Factor'))
        
        st.subheader("AI Recommendations")
        with st.spinner("Generating recommendations..."):
            recommendations = service.get_explanation(user_data, assessment)
            st.write(recommendations)

else:
    st.subheader("Model Training")

    # Session state to persist model/data
    if "trained" not in st.session_state:
        st.session_state.trained = False
        st.session_state.training_data = None

    # Controls
    col1, col2 = st.columns(2)

    with col1:
        n_samples = st.slider("Training Dataset Size", 100, 5000, 500)

    with col2:
        show_data = st.checkbox("Preview Training Data")

    # Generate data button
    if st.button("Generate New Dataset"):
        st.session_state.training_data = generate_data(n_samples)
        st.session_state.trained = False
        st.success(f"Generated {n_samples} samples")

    # Train button
    if st.button("Train Model"):
        if st.session_state.training_data is None:
            st.warning("Generate data first")
        else:
            with st.spinner("Training model..."):
                service.train(st.session_state.training_data)
                st.session_state.trained = True
            st.success("Model trained successfully!")

    # Show data
    if show_data and st.session_state.training_data is not None:
        st.dataframe(st.session_state.training_data.head())

    # After training → show insights
    if st.session_state.trained:
        st.subheader("Model Insights")

        df = st.session_state.training_data

        # Simple feature correlations (proxy insight)
        corr = df.corr(numeric_only=True)

        st.write("Feature Correlation Heatmap")
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

        # Optional: distribution
        st.subheader("Target Distribution")
        if "default" in df.columns:
            st.bar_chart(df["default"].value_counts())
