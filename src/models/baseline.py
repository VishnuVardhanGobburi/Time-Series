from typing import Optional
import numpy as np
import pandas as pd
from src.config import SEASON_LENGTH

class NaiveModel:
    name = "Naive"

    def fit(self, y: pd.Series):
        self.last_value = y.iloc[-1]
        return self

    def predict(self, steps: int) -> np.ndarray:
        return np.repeat(self.last_value, steps)


class SeasonalNaiveModel:
    name = "SeasonalNaive"

    def __init__(self, season_length: int = SEASON_LENGTH):
        self.season_length = season_length

    def fit(self, y: pd.Series):
        if len(y) < self.season_length:
            self.season_values = y.values
        else:
            self.season_values = y.iloc[-self.season_length :].values
        return self

    def predict(self, steps: int) -> np.ndarray:
        if len(self.season_values) == 0:
            return np.zeros(steps)
        repeats = int(np.ceil(steps / len(self.season_values)))
        return np.tile(self.season_values, repeats)[:steps]
