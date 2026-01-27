from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_FREQ, EXPECTED_COLUMNS, OPTIONAL_COLUMNS
from src.data_loader import drop_unnamed
from src.utils import get_logger

LOGGER = get_logger(__name__)


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    expected = {EXPECTED_COLUMNS["date"], EXPECTED_COLUMNS["target"], *EXPECTED_COLUMNS["drivers"]}
    missing = sorted(expected - set(df.columns))
    return len(missing) == 0, missing


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[EXPECTED_COLUMNS["date"]] = pd.to_datetime(
        df[EXPECTED_COLUMNS["date"]], errors="coerce", dayfirst=True
    )
    numeric_cols = [EXPECTED_COLUMNS["target"], *EXPECTED_COLUMNS["drivers"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def enforce_monthly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(EXPECTED_COLUMNS["date"])
    df = df.drop_duplicates(subset=[EXPECTED_COLUMNS["date"]])
    df = df.set_index(EXPECTED_COLUMNS["date"]).asfreq(DEFAULT_FREQ)
    df = df.reset_index()
    return df


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean() * 100.0,
        }
    )
    return report.reset_index().rename(columns={"index": "column"})


def outlier_report(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        rows.append(
            {
                "column": col,
                "iqr_lower": float(lower),
                "iqr_upper": float(upper),
                "outlier_count": int(outliers),
            }
        )
    return pd.DataFrame(rows)


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    df = drop_unnamed(df)
    ok, missing = validate_schema(df)
    if not ok:
        missing_cols = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_cols}")
    df = coerce_types(df)
    df = df.dropna(subset=[EXPECTED_COLUMNS["date"]])
    df = enforce_monthly_frequency(df)
    df = df.sort_values(EXPECTED_COLUMNS["date"]).reset_index(drop=True)
    report = {
        "missingness": missingness_report(df),
        "outliers": outlier_report(df, [EXPECTED_COLUMNS["target"], *EXPECTED_COLUMNS["drivers"]]),
    }
    LOGGER.info("Validation complete with shape %s", df.shape)
    return df, report
