# Finance Risk Python Project

A comprehensive financial risk assessment system combining machine learning, agentic workflows, and scenario analysis.

## Project Structure

- **app/**: Core application modules
  - **main.py**: FastAPI entrypoint
  - **ml/**: Machine learning components (model, feature engineering, SHAP explanations)
  - **agent/**: AI agent with RAG, tools, and orchestration
  - **data/**: Data generation and sample datasets
  - **services/**: Risk scoring and simulation services

- **frontend/**: Streamlit dashboard
- **notebooks/**: Exploratory analysis

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```bash
   ./run.sh
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
