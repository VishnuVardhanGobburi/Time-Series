from typing import Optional, Tuple

import pandas as pd

from src.config import EXPECTED_COLUMNS
from src.utils import get_logger

LOGGER = get_logger(__name__)


def load_csv(path: Optional[str] = None, file_obj=None) -> Tuple[pd.DataFrame, str]:
    if path is None and file_obj is None:
        raise ValueError("Provide a file path or upload a CSV file.")
    if file_obj is not None:
        df = pd.read_csv(file_obj)
        source = "uploaded"
    else:
        df = pd.read_csv(path)
        source = path
    LOGGER.info("Loaded CSV from %s with shape %s", source, df.shape)
    return df, source


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def expected_columns() -> list:
    return [EXPECTED_COLUMNS["date"], EXPECTED_COLUMNS["target"], *EXPECTED_COLUMNS["drivers"]]
