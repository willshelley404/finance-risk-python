from fredapi import Fred
import os

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

def get_macro_data():
    return {
        "unemployment": fred.get_series_latest_release("UNRATE").iloc[-1],
        "interest_rate": fred.get_series_latest_release("FEDFUNDS").iloc[-1],
        "inflation": fred.get_series_latest_release("CPIAUCSL").pct_change().iloc[-1]
    }