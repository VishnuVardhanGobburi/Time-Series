from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import EXPECTED_COLUMNS, LAGS, ROLLING_WINDOWS
from src.features import make_supervised_features


class LGBMModel:
    name = "LightGBM"

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.params = params or default_params
        self.model = lgb.LGBMRegressor(**self.params)
        self.feature_columns = None

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = EXPECTED_COLUMNS["target"],
        driver_cols: Optional[List[str]] = None,
        add_interactions: bool = False,
    ):
        driver_cols = driver_cols or []
        features = make_supervised_features(
            df,
            target_col=target_col,
            driver_cols=driver_cols,
            add_interactions=add_interactions,
            lags=LAGS,
            windows=ROLLING_WINDOWS,
        )
        features = features.dropna()
        self.feature_columns = [
            col for col in features.columns if col not in [EXPECTED_COLUMNS["date"], target_col]
        ]
        X = features[self.feature_columns]
        y = features[target_col]
        self.train_features_ = X
        self.model.fit(X, y)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.feature_columns]
        return self.model.predict(X)


def recursive_forecast(
    model: LGBMModel,
    history: pd.DataFrame,
    future_exog: pd.DataFrame,
    target_col: str,
    driver_cols: List[str],
    horizon: int,
    add_interactions: bool = False,
) -> pd.DataFrame:
    hist = history.copy()
    forecasts = []
    for step in range(horizon):
        current = pd.concat([hist, future_exog.iloc[[step]]], ignore_index=True)
        features = make_supervised_features(
            current,
            target_col=target_col,
            driver_cols=driver_cols,
            add_interactions=add_interactions,
            lags=LAGS,
            windows=ROLLING_WINDOWS,
        ).iloc[[-1]]
        prediction = model.predict(features)[0]
        next_row = future_exog.iloc[[step]].copy()
        next_row[target_col] = prediction
        hist = pd.concat([hist, next_row], ignore_index=True)
        forecasts.append(prediction)
    result = future_exog.copy()
    result[target_col] = forecasts
    return result
