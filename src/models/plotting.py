from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf


def line_plot(df: pd.DataFrame, x: str, y: str, title: str):
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_layout(template="plotly_white")
    return fig


def residual_plot(residuals: np.ndarray, title: str = "Residuals"):
    fig = px.line(y=residuals, title=title)
    fig.update_layout(template="plotly_white", yaxis_title="Residual")
    return fig


def error_distribution(errors: np.ndarray, title: str = "Error Distribution"):
    fig = px.histogram(errors, nbins=30, title=title)
    fig.update_layout(template="plotly_white")
    return fig


def acf_plot(series: np.ndarray, lags: int = 24):
    fig = plot_acf(series, lags=lags, alpha=0.05)
    fig.tight_layout()
    return fig


def forecast_plot(df: pd.DataFrame, date_col: str, actual_col: str, pred_col: str, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], name="Actual"))
    fig.add_trace(go.Scatter(x=df[date_col], y=df[pred_col], name="Forecast"))
    fig.update_layout(template="plotly_white", title=title)
    return fig
