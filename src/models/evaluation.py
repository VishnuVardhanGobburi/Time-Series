from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import EXPECTED_COLUMNS
from src.features import make_supervised_features
from src.models.baseline import NaiveModel, SeasonalNaiveModel
from src.models.ets import ETSModel
from src.models.lgbm import LGBMModel, recursive_forecast
from src.models.sarimax import SARIMAXModel
from src.models.sarimax import sarimax_grid_search
from src.utils import apply_transformation, directional_accuracy, invert_transformation, safe_smape


@dataclass
class FoldResult:
    fold: int
    model: str
    mae: float
    rmse: float
    smape: float
    directional_accuracy: float


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "smape": safe_smape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


def _split_folds(n_obs: int, horizon: int, folds: int) -> List[Tuple[int, int]]:
    min_train = n_obs - horizon * folds
    if min_train <= 0:
        raise ValueError("Not enough data for the requested folds and horizon.")
    splits = []
    for i in range(folds):
        train_end = min_train + i * horizon
        test_end = train_end + horizon
        splits.append((train_end, test_end))
    return splits


def walk_forward_cv(
    df: pd.DataFrame,
    models: List[str],
    horizon: int,
    folds: int,
    exog_cols: Optional[List[str]] = None,
    add_interactions: bool = False,
    transform: str = "none",
    sarimax_grid: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, np.ndarray]]]:
    exog_cols = exog_cols or []
    results = []
    predictions = {}
    n_obs = len(df)
    splits = _split_folds(n_obs, horizon, folds)
    target_col = EXPECTED_COLUMNS["target"]
    df_model = df.copy()
    transformed, _ = apply_transformation(df_model[target_col], transform)
    df_model[target_col] = transformed

    for fold_idx, (train_end, test_end) in enumerate(splits, start=1):
        train = df_model.iloc[:train_end].copy()
        test = df_model.iloc[train_end:test_end].copy()
        y_train_raw = df.iloc[:train_end][target_col]
        y_test_raw = df.iloc[train_end:test_end][target_col].values
        y_train, state = apply_transformation(y_train_raw, transform)
        y_train = y_train.dropna()
        y_test = y_test_raw
        for model_name in models:
            if model_name == "Naive":
                model = NaiveModel().fit(y_train)
                pred_transformed = model.predict(len(test))
                y_pred = invert_transformation(pred_transformed, state)
            elif model_name == "SeasonalNaive":
                model = SeasonalNaiveModel().fit(y_train)
                pred_transformed = model.predict(len(test))
                y_pred = invert_transformation(pred_transformed, state)
            elif model_name == "ETS":
                model = ETSModel().fit(y_train)
                pred_transformed = model.predict(len(test))
                y_pred = invert_transformation(pred_transformed, state)
            elif model_name == "SARIMAX":
                if exog_cols:
                    exog_train = df.loc[y_train.index, exog_cols]
                    exog_test = df.iloc[train_end:test_end][exog_cols]
                else:
                    exog_train = None
                    exog_test = None
                if sarimax_grid:
                    search = sarimax_grid_search(y_train, exog_train, sarimax_grid)
                    result = search["result"]
                    if result is not None:
                        pred_transformed = result.get_forecast(
                            steps=len(test), exog=exog_test
                        ).predicted_mean.values
                    else:
                        model = SARIMAXModel().fit(y_train, exog=exog_train)
                        pred_transformed, _ = model.predict(len(test), exog_future=exog_test)
                else:
                    model = SARIMAXModel().fit(y_train, exog=exog_train)
                    pred_transformed, _ = model.predict(len(test), exog_future=exog_test)
                y_pred = invert_transformation(pred_transformed, state)
            elif model_name == "LightGBM":
                model = LGBMModel().fit(
                    train,
                    target_col=target_col,
                    driver_cols=exog_cols,
                    add_interactions=add_interactions,
                )
                future_exog = df.iloc[train_end:test_end][[EXPECTED_COLUMNS["date"], *exog_cols]]
                forecast_df = recursive_forecast(
                    model,
                    history=train[[EXPECTED_COLUMNS["date"], target_col, *exog_cols]],
                    future_exog=future_exog,
                    target_col=target_col,
                    driver_cols=exog_cols,
                    horizon=len(test),
                    add_interactions=add_interactions,
                )
                pred_transformed = forecast_df[target_col].values
                y_pred = invert_transformation(pred_transformed, state)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            metric = _metrics(y_test, y_pred)
            results.append(
                FoldResult(
                    fold=fold_idx,
                    model=model_name,
                    mae=metric["mae"],
                    rmse=metric["rmse"],
                    smape=metric["smape"],
                    directional_accuracy=metric["directional_accuracy"],
                ).__dict__
            )
            predictions.setdefault(model_name, {})[fold_idx] = y_pred
    return pd.DataFrame(results), predictions
