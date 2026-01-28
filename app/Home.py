import sys
import streamlit as st
from src.config import APP_NAME

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

st.set_page_config(page_title=APP_NAME, layout="wide")

st.title(APP_NAME)
st.subheader("Macro-driven time-series forecasting for retail demand")

st.markdown(
    """
This app forecasts monthly retail sales using classic time-series models and modern ML,
augmented with macroeconomic drivers. It provides diagnostics, walk-forward validation,
explainability, and business-ready insights.
"""
)

st.markdown("### What this project does")
st.markdown(
    """
- Loads `sales.csv` (or user upload) and validates the schema
- Explores trend/seasonality and driver relationships
- Trains baseline, ETS, SARIMAX, and LightGBM models
- Evaluates with walk-forward CV and picks the best model
- Explains driver impact with coefficients and SHAP
"""
)

st.markdown("### Why this matters for business")
st.markdown(
    """
Retail demand is sensitive to macro conditions. A forecast enriched with CPI, unemployment,
interest rates, and sentiment helps teams plan inventory, staffing, and marketing with
greater confidence.
"""
)

st.markdown("### Quick start")
st.markdown(
    """
1. Go to **Data Explorer** and load `sales.csv` or upload your file.
2. Inspect schema checks, missingness, and time-series coverage.
3. Use **Modeling** to run cross-validation and select a model.
4. Use **Forecast & Explain** for scenario analysis and exports.
"""
)


