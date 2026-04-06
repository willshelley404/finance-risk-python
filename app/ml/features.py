import pandas as pd
import numpy as np

class FeatureEngineer:
    @staticmethod
    def create_features(df):
        df_features = df.copy()
        
        # Financial ratios
        df_features['debt_to_income'] = df['total_debt'] / (df['monthly_income'].replace(0, 1))
        df_features['savings_ratio'] = df['savings'] / (df['monthly_income'] + 1)
        df_features['expense_ratio'] = df['monthly_expenses'] / (df['monthly_income'] + 1)
        
        # Cash flow indicators
        df_features['monthly_cashflow'] = df['monthly_income'] - df['monthly_expenses']
        df_features['emergency_fund_months'] = df['savings'] / (df['monthly_expenses'] + 1)
        
        # Credit indicators
        df_features['credit_utilization'] = df['credit_card_balance'] / (df['credit_limit'].replace(0, 1))

        
        # Payment history (late payments in last 12 months)
        df_features['late_payment_ratio'] = df['late_payments_12m'] / 12
        
        return df_features[['debt_to_income', 'savings_ratio', 'expense_ratio', 
                          'monthly_cashflow', 'emergency_fund_months', 
                          'credit_utilization', 'late_payment_ratio', 'age', 'credit_score']]
