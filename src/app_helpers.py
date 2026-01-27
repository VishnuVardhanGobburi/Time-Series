from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from src.data_loader import load_csv
from src.validation import validate_and_clean


@st.cache_data(show_spinner=False)
def load_and_validate(path: Optional[str] = None, file_obj=None) -> Tuple[pd.DataFrame, dict, str]:
    df_raw, source = load_csv(path=path, file_obj=file_obj)
    df, report = validate_and_clean(df_raw)
    return df, report, source
