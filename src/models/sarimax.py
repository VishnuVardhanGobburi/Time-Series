from itertools import product
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAXModel:
    name = "SARIMAX"

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.fit_result = None

    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self.fit_result = self.model.fit(disp=False)
        return self

    def predict(self, steps: int, exog_future: Optional[pd.DataFrame] = None):
        forecast_res = self.fit_result.get_forecast(steps=steps, exog=exog_future)
        mean = forecast_res.predicted_mean.values
        conf = forecast_res.conf_int()
        return mean, conf


def sarimax_grid_search(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    param_grid: Dict[str, Tuple[int, ...]],
) -> Dict:
    best_aic = np.inf
    best_params = None
    best_result = None
    orders = list(product(param_grid["p"], param_grid["d"], param_grid["q"]))
    seasonal_orders = list(
        product(param_grid["P"], param_grid["D"], param_grid["Q"], param_grid["s"])
    )
    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                model = SARIMAX(
                    y,
                    exog=exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_params = {"order": order, "seasonal_order": seasonal_order}
                    best_result = result
            except Exception:
                continue
    return {"params": best_params, "result": best_result, "aic": best_aic}
