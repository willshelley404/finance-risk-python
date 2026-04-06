import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Model settings
MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_RISK_THRESHOLD = 0.5

# FRED Economic Indicators
FRED_INDICATORS = {
    "unemployment": "UNRATE",
    "inflation": "CPIAUCSL",
    "prime_rate": "TERMCBCCALLNS",
    "consumer_debt": "TOTALSL",
}
