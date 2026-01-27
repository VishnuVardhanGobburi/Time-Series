import json
import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.app_helpers import load_and_validate
from src.config import DEFAULT_HORIZON, EXPECTED_COLUMNS
from src.features import make_future_frame
from src.models.baseline import NaiveModel, SeasonalNaiveModel
from src.models.ets import ETSModel
from src.models.explainability import driver_story, sarimax_coefficients, shap_summary_plot
from src.models.lgbm import LGBMModel, recursive_forecast
from src.models.plotting import acf_plot, error_distribution, forecast_plot, residual_plot
from src.models.sarimax import SARIMAXModel
from src.utils import apply_transformation, invert_transformation, ks_drift_test, zscore_shift
from src.utils import safe_smape, directional_accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Forecast & Explain", layout="wide")
st.title("Forecast & Explain")

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
driver_cols = EXPECTED_COLUMNS["drivers"]

st.subheader("Configuration")
model_name = st.selectbox(
    "Model",
    ["Naive", "SeasonalNaive", "ETS", "SARIMAX", "LightGBM"],
    index=3,
)
use_exog = st.checkbox("Use exogenous drivers", value=True)
transform = st.selectbox("Transformation", ["none", "log", "diff", "pct_change"], index=0)
sarimax_auto = st.checkbox("SARIMAX small grid search", value=False)
train_end = st.date_input(
    "Train end date",
    value=df[date_col].iloc[-DEFAULT_HORIZON].date(),
    min_value=df[date_col].min().date(),
    max_value=df[date_col].max().date(),
)
horizon = st.slider("Forecast horizon (months)", 1, 24, DEFAULT_HORIZON)

train_end_ts = pd.to_datetime(train_end)
train = df[df[date_col] <= train_end_ts].copy()
test = df[df[date_col] > train_end_ts].copy()
cols_required = [target_col] + (driver_cols if use_exog else [])
train = train.dropna(subset=cols_required)
test = test.dropna(subset=cols_required)

if len(test) < 1:
    st.warning("Train end date must leave at least one month for testing.")
    st.stop()

st.subheader("Scenario forecasting")
st.caption("Adjust macro drivers for a what-if scenario.")
scenario_adjustments = {}
for col in driver_cols:
    scenario_adjustments[col] = st.slider(
        f"{col} adjustment (%)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
    )

future_exog = pd.DataFrame()
if use_exog:
    last_values = train[driver_cols].iloc[-1]
    scenario = {}
    for col in driver_cols:
        scenario[col] = last_values[col] * (1.0 + scenario_adjustments[col] / 100.0)
    scenario_df = pd.DataFrame([scenario] * horizon)
    scenario_df[date_col] = pd.date_range(
        start=train[date_col].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    future_exog = scenario_df[[date_col] + driver_cols]

st.subheader("Model training and forecast")
with st.spinner("Training model..."):
    y_train_raw = train[target_col]
    y_train_transformed, state = apply_transformation(y_train_raw, transform)

    forecast = None
    conf_int = None

    if model_name == "Naive":
        model = NaiveModel().fit(y_train_transformed)
        pred_trans = model.predict(horizon)
        forecast = invert_transformation(pred_trans, state)
    elif model_name == "SeasonalNaive":
        model = SeasonalNaiveModel().fit(y_train_transformed)
        pred_trans = model.predict(horizon)
        forecast = invert_transformation(pred_trans, state)
    elif model_name == "ETS":
        model = ETSModel().fit(y_train_transformed)
        pred_trans = model.predict(horizon)
        forecast = invert_transformation(pred_trans, state)
    elif model_name == "SARIMAX":
        exog_train = train[driver_cols] if use_exog else None
        if sarimax_auto:
            from src.models.sarimax import sarimax_grid_search

            grid = {"p": (0, 1), "d": (0, 1), "q": (0, 1), "P": (0, 1), "D": (0, 1), "Q": (0, 1), "s": (12,)}
            search = sarimax_grid_search(y_train_transformed, exog_train, grid)
            result = search["result"]
            exog_future = future_exog[driver_cols] if use_exog else None
            if result is not None:
                pred_trans = result.get_forecast(steps=horizon, exog=exog_future).predicted_mean.values
                forecast = invert_transformation(pred_trans, state)
            else:
                model = SARIMAXModel().fit(y_train_transformed, exog=exog_train)
                pred_trans, conf = model.predict(horizon, exog_future=exog_future)
                forecast = invert_transformation(pred_trans, state)
        else:
            model = SARIMAXModel().fit(y_train_transformed, exog=exog_train)
            exog_future = future_exog[driver_cols] if use_exog else None
            pred_trans, conf = model.predict(horizon, exog_future=exog_future)
            forecast = invert_transformation(pred_trans, state)
            if transform in ["none", "log"]:
                conf_int = conf
                if transform == "log":
                    conf_int = np.exp(conf_int)
    elif model_name == "LightGBM":
        train_model = train.copy()
        train_model[target_col] = y_train_transformed
        model = LGBMModel().fit(train_model, target_col=target_col, driver_cols=driver_cols if use_exog else [])
        if use_exog:
            future = future_exog.copy()
        else:
            future = make_future_frame(train[date_col].iloc[-1], horizon)
            for col in driver_cols:
                future[col] = train[col].iloc[-1]
        forecast_df = recursive_forecast(
            model,
            history=train_model[[date_col, target_col] + (driver_cols if use_exog else [])],
            future_exog=future[[date_col] + (driver_cols if use_exog else [])],
            target_col=target_col,
            driver_cols=driver_cols if use_exog else [],
            horizon=horizon,
        )
        forecast = invert_transformation(forecast_df[target_col].values, state)

forecast_dates = pd.date_range(
    start=train[date_col].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
)
forecast_df = pd.DataFrame({date_col: forecast_dates, "forecast": forecast})
if conf_int is not None and len(conf_int) == horizon:
    forecast_df["lower"] = conf_int.iloc[:, 0].values
    forecast_df["upper"] = conf_int.iloc[:, 1].values

st.subheader("Forecast result")
plot_df = pd.concat(
    [
        train[[date_col, target_col]].rename(columns={target_col: "actual"}),
        test[[date_col, target_col]].rename(columns={target_col: "actual"}),
    ],
    ignore_index=True,
)
plot_df = plot_df.merge(forecast_df, on=date_col, how="left")
fig = forecast_plot(plot_df, date_col, "actual", "forecast", "Actual vs Forecast")
st.plotly_chart(fig, use_container_width=True)

if "lower" in forecast_df.columns:
    st.dataframe(forecast_df, use_container_width=True)

st.subheader("Diagnostics")
if len(test) >= 2:
    y_true = test[target_col].iloc[:horizon].values
    y_pred = forecast[: len(y_true)]
    residuals = y_true - y_pred
    st.plotly_chart(residual_plot(residuals), use_container_width=True)
    st.plotly_chart(error_distribution(residuals), use_container_width=True)
    st.pyplot(acf_plot(residuals))
    lb = acorr_ljungbox(residuals, lags=[12], return_df=True)
    st.write({"Ljung-Box p-value (lag 12)": float(lb["lb_pvalue"].iloc[0])})
    eval_metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "smape": safe_smape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
    st.write(eval_metrics)

st.subheader("Explainability")
if model_name == "SARIMAX" and use_exog:
    coeffs = sarimax_coefficients(model.fit_result)
    st.dataframe(coeffs, use_container_width=True)
    st.markdown("**Driver story**")
    for line in driver_story(coeffs):
        st.write(f"- {line}")

if model_name == "LightGBM":
    try:
        features = model.feature_columns
        st.write("Top model features:", features[:10])
        fig = shap_summary_plot(model.model, model.train_features_)
        st.pyplot(fig)
    except Exception:
        st.warning("SHAP summary unavailable for this run.")

st.subheader("What changed? (Last 6 vs prior 6 months)")
window = 6
recent = df.tail(window)[target_col]
prior = df.tail(window * 2).head(window)[target_col]
growth_recent = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100
growth_prior = (prior.iloc[-1] - prior.iloc[0]) / prior.iloc[0] * 100
vol_recent = recent.std()
vol_prior = prior.std()
st.write(
    {
        "recent_growth_pct": growth_recent,
        "prior_growth_pct": growth_prior,
        "recent_volatility": float(vol_recent),
        "prior_volatility": float(vol_prior),
    }
)

st.subheader("Business interpretation")
trend_direction = "increase" if forecast[-1] > train[target_col].iloc[-1] else "soften"
st.write(f"- Forecast indicates demand may {trend_direction} over the next {horizon} months.")
st.write(f"- Recent growth ({growth_recent:.2f}%) vs prior ({growth_prior:.2f}%) highlights momentum shifts.")
if use_exog:
    st.write("- Scenario adjustments reflect how macro changes could move sales expectations.")

st.subheader("Model monitoring")
monitor_window = st.slider("Monitoring window (months)", 6, 24, 12)
train_hist = df.iloc[:-monitor_window][target_col]
recent_hist = df.iloc[-monitor_window:][target_col]
drift_stats = {
    "ks_test": ks_drift_test(train_hist.values, recent_hist.values),
    "zscore_shift": zscore_shift(train_hist.values, recent_hist.values),
}
st.write(drift_stats)
if drift_stats["ks_test"]["p_value"] < 0.05 or abs(drift_stats["zscore_shift"]["mean_shift"]) > 1.0:
    st.warning("Potential drift detected. Consider retraining or recalibrating.")

st.subheader("Export outputs")
metrics = {
    "model": model_name,
    "transform": transform,
    "horizon": horizon,
}
if len(test) >= 2:
    metrics.update(eval_metrics)
st.download_button(
    "Download forecast CSV",
    forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="forecast.csv",
    mime="text/csv",
)
st.download_button(
    "Download metrics JSON",
    json.dumps(metrics, indent=2).encode("utf-8"),
    file_name="metrics.json",
    mime="application/json",
)
