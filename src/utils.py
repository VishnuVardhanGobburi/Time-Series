import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import LOGGING_FORMAT


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    return logging.getLogger(name)


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1.0, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_change = np.sign(np.diff(y_true))
    pred_change = np.sign(np.diff(y_pred))
    if len(true_change) == 0:
        return np.nan
    return np.mean(true_change == pred_change) * 100.0


@dataclass
class TransformState:
    name: str
    last_value: Optional[float] = None


def apply_transformation(series: pd.Series, transform: str) -> Tuple[pd.Series, TransformState]:
    if transform == "none":
        return series.copy(), TransformState(name="none")
    if transform == "log":
        return np.log(series.replace(0, np.nan)).dropna(), TransformState(name="log")
    if transform == "diff":
        return series.diff().dropna(), TransformState(name="diff", last_value=series.iloc[-1])
    if transform == "pct_change":
        return series.pct_change().dropna(), TransformState(name="pct_change", last_value=series.iloc[-1])
    raise ValueError(f"Unknown transformation: {transform}")


def invert_transformation(
    transformed: Iterable[float], state: TransformState
) -> np.ndarray:
    transformed = np.asarray(list(transformed), dtype=float)
    if state.name == "none":
        return transformed
    if state.name == "log":
        return np.exp(transformed)
    if state.name == "diff":
        if state.last_value is None:
            raise ValueError("Missing last_value for diff inversion.")
        return np.r_[state.last_value, transformed].cumsum()[1:]
    if state.name == "pct_change":
        if state.last_value is None:
            raise ValueError("Missing last_value for pct_change inversion.")
        values = []
        prev = state.last_value
        for pct in transformed:
            prev = prev * (1.0 + pct)
            values.append(prev)
        return np.array(values)
    raise ValueError(f"Unknown transformation state: {state.name}")


def to_json(data: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def monthly_date_range(start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="MS")


def ks_drift_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    stat, pvalue = stats.ks_2samp(a, b, nan_policy="omit")
    return {"ks_stat": float(stat), "p_value": float(pvalue)}


def zscore_shift(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a_mean, a_std = np.nanmean(a), np.nanstd(a)
    b_mean, b_std = np.nanmean(b), np.nanstd(b)
    if a_std == 0 or np.isnan(a_std):
        return {"mean_shift": float("nan"), "std_shift": float("nan")}
    return {
        "mean_shift": float((b_mean - a_mean) / a_std),
        "std_shift": float((b_std - a_std) / a_std),
    }
