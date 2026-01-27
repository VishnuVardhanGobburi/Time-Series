APP_NAME = "Retail Sales Forecasting with Macro Drivers"

EXPECTED_COLUMNS = {
    "date": "date",
    "target": "retail_sales",
    "drivers": [
        "cpi_value",
        "Unemployement_Rate",
        "Interest_Rate",
        "Consumer_Sentiment",
    ],
}

OPTIONAL_COLUMNS = {"month", "year"}

DEFAULT_TEST_SIZE = 12
DEFAULT_FREQ = "MS"  # Month start

LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]

DEFAULT_CV_FOLDS = 4
DEFAULT_HORIZON = 12

SEASON_LENGTH = 12

LOGGING_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
