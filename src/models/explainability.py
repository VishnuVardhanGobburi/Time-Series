from typing import Dict, List

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import EXPECTED_COLUMNS


def sarimax_coefficients(result) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    summary = pd.DataFrame(
        {
            "coef": params,
            "ci_lower": conf.iloc[:, 0],
            "ci_upper": conf.iloc[:, 1],
        }
    )
    summary.index.name = "feature"
    return summary.reset_index()


def shap_summary(model, features: pd.DataFrame, max_samples: int = 500) -> Dict[str, np.ndarray]:
    sample = features.sample(min(len(features), max_samples), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return {"sample": sample, "shap_values": shap_values}


def shap_summary_plot(model, features: pd.DataFrame, max_samples: int = 500):
    sample = features.sample(min(len(features), max_samples), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    fig = plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    return fig


def driver_story(coeff_table: pd.DataFrame) -> List[str]:
    stories = []
    for _, row in coeff_table.iterrows():
        feature = row["feature"]
        if feature in EXPECTED_COLUMNS["drivers"]:
            direction = "increases" if row["coef"] > 0 else "decreases"
            stories.append(
                f"Retail sales tends to {direction} when {feature.replace('_', ' ')} rises."
            )
    return stories
