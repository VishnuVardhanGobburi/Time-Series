import pandas as pd
import pytest

from src.validation import validate_and_clean


def test_validate_and_clean_drops_unnamed_and_sorts():
    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "date": ["01-02-2020", "01-01-2020"],
            "retail_sales": [100, 110],
            "cpi_value": [1.0, 1.1],
            "Unemployement_Rate": [5.0, 5.1],
            "Interest_Rate": [1.0, 1.1],
            "Consumer_Sentiment": [80, 81],
        }
    )
    cleaned, report = validate_and_clean(df)
    assert "Unnamed: 0" not in cleaned.columns
    assert cleaned["date"].is_monotonic_increasing
    assert report["missingness"].shape[0] >= 1


def test_validate_missing_columns():
    df = pd.DataFrame({"date": ["2020-01-01"], "retail_sales": [100]})
    with pytest.raises(ValueError):
        validate_and_clean(df)
