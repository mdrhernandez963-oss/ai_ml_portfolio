import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="House Price Error Analysis",
    layout="wide"
)

# -------------------------------
# Paths
# -------------------------------
#import os 
#from pathlib import Path
#BASE_DIR = Path(__file__).resolve().parent
#MODEL_DIR = BASE_DIR.parents[2] / "models"  # ai-portfolio/models
#ARTIFACT_PATH = MODEL_DIR / "California_house_value_error_analysis.pk1"
#BASE_DIR = Path(os.getcwd())  # instead of __file__
#MODEL_DIR = BASE_DIR / "models"
#ARTIFACT_PATH = MODEL_DIR / "California_house_value_error_analysis.pk1"
from pathlib import Path

# Current app folder
BASE_DIR = Path(__file__).resolve().parent

# Go up from app folder to ai-portfolio
MODEL_DIR = BASE_DIR.parents[1] / "models"  # adjust number of .parents
ARTIFACT_PATH = MODEL_DIR / "California_house_value_error_analysis.pk1"

print(ARTIFACT_PATH)
print(ARTIFACT_PATH.exists())  # should be True if path is correct

# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACT_PATH)

if not ARTIFACT_PATH.exists():
    st.error(f"❌ Artifact file not found:\n{ARTIFACT_PATH}")
    st.stop()

artifacts = load_artifacts()

X_test = artifacts["X_test"]
y_true = artifacts["y_true"]
y_pred = artifacts["y_pred"]
residuals = artifacts["residuals"]
metrics = artifacts["metrics"]

# -------------------------------
# Header
# -------------------------------
st.title("🏠 California House Price — Error Analysis Dashboard")
st.caption("Explore model performance using historical test data")

st.markdown(
    """
    This dashboard examines:
    - Systematic underprediction of high-value homes
    - Geographic error patterns
    - High-risk predictions that could mislead decisions
    """
)

# -------------------------------
# Metrics
# -------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"${metrics['mae']:,.0f}")
col2.metric("RMSE", f"${metrics['rmse']:,.0f}")
col3.metric("R²", f"{metrics['r2']:.3f}")

st.divider()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 High-Value Home Prediction Errors",
    "🌍 Geographic Bias",
    "⚠️ High-Risk Errors"
])

# ============================================================
# Tab 1: High-Value Home Prediction Errors
# ============================================================
with tab1:
    st.subheader("Does the model underpredict expensive homes?")
    
    st.markdown(
        """
        **Related plot:** Residuals for high-value properties
        
        - Prediction errors are larger and more negative for expensive homes.
        - The model systematically underpredicts premium properties.
        - Large negative residuals represent the **highest potential financial impact**.
        - Users should interpret predictions for high-value homes with caution.
        """
    )

    price_bins = pd.qcut(y_true, q=3, labels=["Low", "Mid", "High"])
    df_price = pd.DataFrame({
        "Residual": residuals,
        "Price Segment": price_bins
    })

    fig, ax = plt.subplots()
    df_price.boxplot(column="Residual", by="Price Segment", ax=ax)
    ax.axhline(0, color="red")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Distribution by Price Segment")
    plt.suptitle("")
    st.pyplot(fig)

    st.info(
        "Conclusion: High-value homes show larger and more frequent negative residuals, "
        "indicating systematic underprediction. These errors carry higher financial risk than small errors in lower-priced homes, "
        "and may point to missing features like luxury amenities or location desirability."
    )

# ============================================================
# Tab 2: Geographic Bias
# ============================================================
with tab2:
    st.subheader("Are prediction errors worse in certain geographic regions?")
    
    st.markdown(
        """
        **Related plot:** Geographic scatter plot of prediction errors
        
        Each point represents a home. Color indicates the residual magnitude.
        """
    )

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X_test["longitude"],
        X_test["latitude"],
        c=residuals,
        alpha=0.6,
        cmap="coolwarm"
    )
    plt.colorbar(scatter, ax=ax, label="Residual")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of Prediction Errors")
    st.pyplot(fig)

    st.info(
        "Conclusion: Errors cluster geographically, with higher-magnitude errors in "
        "coastal and dense urban regions, suggesting location-based bias or missing features."
    )

# ============================================================
# Tab 3: High-Risk Errors
# ============================================================
with tab3:
    st.subheader("Which prediction errors would cause the most harm?")
    
    st.markdown(
        """
        **Related table:** Top absolute errors
        
        Focus on large residuals that could mislead decision-making.
        """
    )

    df_risk = X_test.copy()
    df_risk["Actual Price"] = y_true
    df_risk["Predicted Price"] = y_pred
    df_risk["Residual"] = residuals
    df_risk["Absolute Error"] = np.abs(residuals)

    top_n = st.slider("Show top N highest-risk predictions", 5, 50, 10)

    worst_cases = df_risk.sort_values("Absolute Error", ascending=False).head(top_n)

    st.dataframe(
        worst_cases[[
            "Actual Price",
            "Predicted Price",
            "Residual",
            "Absolute Error",
            "latitude",
            "longitude"
        ]],
        use_container_width=True
    )

    st.warning(
        "Conclusion: Large underpredictions on high-value homes represent the highest financial risk "
        "and warrant special attention when interpreting model predictions."
    )

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption(
    "This dashboard is for diagnostic and decision-support purposes only."
)

