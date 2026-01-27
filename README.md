# Retail Sales Forecasting with Macro Drivers

## Project summary
This project delivers a production-style Streamlit app for monthly retail sales forecasting
using macroeconomic drivers (CPI, unemployment, interest rate, and consumer sentiment). It
includes data validation, diagnostics, walk-forward cross-validation, multiple model
comparisons (baseline, ETS, SARIMAX, LightGBM), explainability, and scenario analysis.

## Repository structure
```
app/
  Home.py
  pages/
    1_Data_Explorer.py
    2_EDA.py
    3_Modeling.py
    4_Forecast_Explain.py
src/
  config.py
  app_helpers.py
  data_loader.py
  validation.py
  features.py
  models/
    baseline.py
    ets.py
    sarimax.py
    lgbm.py
    evaluation.py
    explainability.py
    plotting.py
  utils.py
tests/
  test_validation.py
  test_features.py
sales.csv
requirements.txt
```

## How to run locally
1. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```
   streamlit run app/Home.py
   ```

## Tests
Run unit tests with:
```
pytest
```

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Set the entry point to `app/Home.py`.
4. Ensure `requirements.txt` is detected for dependencies.

## Screenshots (placeholders)
Add screenshots of each page here:
- Overview page
- Data Explorer
- EDA & Diagnostics
- Modeling
- Forecast & Explain

## Modeling and evaluation methodology
The app uses **walk-forward / expanding-window cross-validation** with a configurable
number of folds (default 4) and forecast horizon. For each fold:
1. Train on all data up to the fold cutoff.
2. Predict the next `horizon` months.
3. Compute MAE, RMSE, sMAPE, and directional accuracy.

Model selection chooses the lowest mean CV sMAPE, using RMSE as a tie-breaker.

## Notes
- The app validates required columns and gracefully errors if missing.
- Monthly frequency is enforced and the data is sorted by date.
- Scenario forecasting allows macro driver adjustments for what-if analysis.
- Model monitoring provides drift checks on recent data.
