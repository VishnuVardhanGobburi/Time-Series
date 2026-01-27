import streamlit as st
from src.app_helpers import load_and_validate
from src.config import EXPECTED_COLUMNS
from src.models.plotting import line_plot


st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")

st.markdown("Load the local `sales.csv` or upload your own dataset.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_local = st.checkbox("Use local sales.csv", value=True)

df = None
report = None
source = None
try:
    if uploaded is not None:
        df, report, source = load_and_validate(file_obj=uploaded)
    elif use_local:
        df, report, source = load_and_validate(path="sales.csv")
except Exception as exc:
    st.error(f"Data validation failed: {exc}")

if df is None:
    st.info("Upload a CSV or enable the local file option.")
    st.stop()

st.success(f"Loaded data from {source} with {len(df)} rows.")

st.subheader("Schema validation")
st.write("Required columns:", [EXPECTED_COLUMNS["date"], EXPECTED_COLUMNS["target"], *EXPECTED_COLUMNS["drivers"]])
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Missingness report")
st.dataframe(report["missingness"], use_container_width=True)

st.subheader("Outlier report (IQR rule)")
st.dataframe(report["outliers"], use_container_width=True)

st.subheader("Time series plot")
fig = line_plot(df, EXPECTED_COLUMNS["date"], EXPECTED_COLUMNS["target"], "Retail Sales")
st.plotly_chart(fig, use_container_width=True)
