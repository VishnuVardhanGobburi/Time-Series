import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from src.app_helpers import load_and_validate
from src.config import EXPECTED_COLUMNS
from src.models.plotting import line_plot


st.set_page_config(page_title="EDA & Diagnostics", layout="wide")
st.title("EDA & Diagnostics")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_local = st.checkbox("Use local sales.csv", value=True)

try:
    if uploaded is not None:
        df, _, _ = load_and_validate(file_obj=uploaded)
    elif use_local:
        df, _, _ = load_and_validate(path="sales.csv")
    else:
        df = None
except Exception as exc:
    st.error(f"Data validation failed: {exc}")
    df = None

if df is None:
    st.stop()

target_col = EXPECTED_COLUMNS["target"]
date_col = EXPECTED_COLUMNS["date"]

st.subheader("Trend and seasonality")
fig = line_plot(df, date_col, target_col, "Retail Sales")
st.plotly_chart(fig, use_container_width=True)

st.subheader("STL decomposition")
stl = STL(df[target_col], period=12)
res = stl.fit()
decomp = pd.DataFrame(
    {
        date_col: df[date_col],
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid,
    }
)
st.plotly_chart(line_plot(decomp, date_col, "trend", "Trend"), use_container_width=True)
st.plotly_chart(line_plot(decomp, date_col, "seasonal", "Seasonality"), use_container_width=True)
st.plotly_chart(line_plot(decomp, date_col, "resid", "Residual"), use_container_width=True)

st.subheader("Stationarity tests")
try:
    adf_p = adfuller(df[target_col].dropna())[1]
    kpss_p = kpss(df[target_col].dropna(), nlags="auto")[1]
    st.write({"ADF p-value": adf_p, "KPSS p-value": kpss_p})
except Exception as exc:
    st.warning(f"Stationarity tests failed: {exc}")

st.subheader("Correlation with macro drivers")
max_lag = st.slider("Max lag (months)", 0, 12, 6)
lagged_corr = {}
for lag in range(0, max_lag + 1):
    shifted = df.copy()
    for driver in EXPECTED_COLUMNS["drivers"]:
        shifted[f"{driver}_lag_{lag}"] = shifted[driver].shift(lag)
    cols = [f"{d}_lag_{lag}" for d in EXPECTED_COLUMNS["drivers"]]
    corr = shifted[[target_col] + cols].corr().iloc[0, 1:]
    lagged_corr[f"lag_{lag}"] = corr
corr_df = pd.DataFrame(lagged_corr)
st.dataframe(corr_df, use_container_width=True)
