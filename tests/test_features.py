import pandas as pd

from src.features import add_calendar_features, add_lag_features, add_rolling_features


def test_feature_generation():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="MS"),
            "retail_sales": [10, 11, 12, 13, 14, 15],
        }
    )
    df = add_calendar_features(df)
    df = add_lag_features(df, "retail_sales", lags=[1, 2])
    df = add_rolling_features(df, "retail_sales", windows=[3])
    assert "month" in df.columns
    assert "retail_sales_lag_1" in df.columns
    assert "retail_sales_roll_mean_3" in df.columns
