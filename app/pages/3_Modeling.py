import streamlit as st
import pandas as pd

from src.app_helpers import load_and_validate
from src.config import DEFAULT_CV_FOLDS, DEFAULT_HORIZON, EXPECTED_COLUMNS
from src.models.evaluation import walk_forward_cv
from src.models.plotting import forecast_plot


st.set_page_config(page_title="Modeling", layout="wide")
st.title("Modeling")

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

model_options = ["Naive", "SeasonalNaive", "ETS", "SARIMAX", "LightGBM"]
selected_models = st.multiselect("Select models", model_options, default=model_options)

use_exog = st.checkbox("Use exogenous macro drivers", value=True)
transform = st.selectbox("Transformation", ["none", "log", "diff", "pct_change"], index=0)
cv_folds = st.slider("CV folds", 3, 6, DEFAULT_CV_FOLDS)
horizon = st.slider("Forecast horizon (months)", 1, 24, DEFAULT_HORIZON)
auto_select = st.checkbox("Auto Select best model", value=True)
sarimax_auto = st.checkbox("SARIMAX small grid search", value=False)

if st.button("Run Modeling"):
    with st.spinner("Running walk-forward CV..."):
        exog_cols = EXPECTED_COLUMNS["drivers"] if use_exog else []
        df_model = df.dropna(subset=[EXPECTED_COLUMNS["target"]] + exog_cols)
        sarimax_grid = None
        if sarimax_auto:
            sarimax_grid = {"p": (0, 1), "d": (0, 1), "q": (0, 1), "P": (0, 1), "D": (0, 1), "Q": (0, 1), "s": (12,)}
        results, predictions = walk_forward_cv(
            df_model,
            models=selected_models,
            horizon=horizon,
            folds=cv_folds,
            exog_cols=exog_cols,
            transform=transform,
            sarimax_grid=sarimax_grid,
        )
    st.subheader("Fold-by-fold metrics")
    st.dataframe(results, use_container_width=True)

    st.subheader("Model ranking")
    summary = (
        results.groupby("model")
        .agg(smape_mean=("smape", "mean"), rmse_mean=("rmse", "mean"))
        .sort_values(["smape_mean", "rmse_mean"])
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)

    if auto_select and not summary.empty:
        best_model = summary.iloc[0]["model"]
        st.success(f"Auto-selected model: {best_model}")

        last_fold = results["fold"].max()
        pred = predictions.get(best_model, {}).get(last_fold)
        if pred is not None:
            test_start = len(df) - horizon
            plot_df = df.iloc[test_start:].copy()
            plot_df["forecast"] = pred[: len(plot_df)]
            fig = forecast_plot(
                plot_df,
                EXPECTED_COLUMNS["date"],
                EXPECTED_COLUMNS["target"],
                "forecast",
                f"Backtest (Fold {last_fold})",
            )
            st.subheader("Backtest plot")
            st.plotly_chart(fig, use_container_width=True)
