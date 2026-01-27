from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.config import EXPECTED_COLUMNS, LAGS, ROLLING_WINDOWS


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[EXPECTED_COLUMNS["date"]].dt.month
    df["year"] = df[EXPECTED_COLUMNS["date"]].dt.year
    return df


def add_lag_features(
    df: pd.DataFrame, target_col: str, lags: Optional[Iterable[int]] = None
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags or LAGS:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str, windows: Optional[Iterable[int]] = None
) -> pd.DataFrame:
    df = df.copy()
    for window in windows or ROLLING_WINDOWS:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].shift(1).rolling(window).std()
    return df


def add_driver_interactions(df: pd.DataFrame, driver_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in driver_cols:
        df[f"{col}_x_month"] = df[col] * df["month"]
    return df


def make_supervised_features(
    df: pd.DataFrame,
    target_col: str = EXPECTED_COLUMNS["target"],
    driver_cols: Optional[List[str]] = None,
    add_interactions: bool = False,
    lags: Optional[Iterable[int]] = None,
    windows: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    driver_cols = driver_cols or []
    df_feat = add_calendar_features(df)
    df_feat = add_lag_features(df_feat, target_col, lags=lags)
    df_feat = add_rolling_features(df_feat, target_col, windows=windows)
    if driver_cols:
        for col in driver_cols:
            if col not in df_feat.columns:
                df_feat[col] = df[col]
    if add_interactions and driver_cols:
        df_feat = add_driver_interactions(df_feat, driver_cols)
    return df_feat


def make_future_frame(
    last_date: pd.Timestamp,
    horizon: int,
    driver_values: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    future = pd.DataFrame({EXPECTED_COLUMNS["date"]: future_dates})
    future = add_calendar_features(future)
    if driver_values is not None:
        for col in driver_values.columns:
            future[col] = driver_values[col].values[:horizon]
    return future
