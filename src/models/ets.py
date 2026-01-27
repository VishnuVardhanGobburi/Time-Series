from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ETSModel:
    name = "ETS"

    def __init__(self, seasonal_periods: int = 12, trend: str = "add", seasonal: str = "add"):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        self.fit_result = None

    def fit(self, y: pd.Series):
        self.model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self.fit_result = self.model.fit(optimized=True)
        return self

    def predict(self, steps: int) -> np.ndarray:
        return self.fit_result.forecast(steps).values
