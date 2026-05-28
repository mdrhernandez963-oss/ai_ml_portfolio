# apps/house_price/house_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import requests

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="California House Price Prediction & Error Analysis",
    layout="wide"
)

# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
@st.cache_resource
def load_house_model():

    url = "https://huggingface.co/Maria123-ai/employment_models/resolve/main/California_house_value_model.pk1"

    response = requests.get(url)

    with open("California_house_value_model.pk1", "wb") as f:
        f.write(response.content)

    return joblib.load("California_house_value_model.pk1")

# ------------------------------------------------------------
# Load ZIP code data
# ------------------------------------------------------------
@st.cache_data
def load_zip_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_csv_path = os.path.join(
        script_dir, "..", "..", "data", "raw", "california_zip_lat_long_full.csv"
    )
    df = pd.read_csv(zip_csv_path)
    df["zip_display"] = df["zip"].astype(str) + " — " + df["city"]
    return df

# ------------------------------------------------------------
# Load error analysis artifacts
# ------------------------------------------------------------
@st.cache_resource
def load_error_artifacts():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_path = os.path.join(
        script_dir, "..", "..", "models", "California_house_value_error_analysis.pk1"
    )

    if not os.path.exists(artifact_path):
        st.error(f"❌ Error analysis artifact not found:\n{artifact_path}")
        st.stop()

    return joblib.load(artifact_path)

# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
def run():

    # ============================================================
    # HOUSE PRICE PREDICTION
    # ============================================================
    st.title("🏠 California House Value Prediction")
    
    model = load_house_model()
    zip_df = load_zip_data()

    st.subheader("Housing Information")

    housing_median_age = st.number_input("House Age (years, 1–60)", 1.0, 60.0, 10.0)
    AveRooms = st.number_input("Average Rooms per Household (1–15)", 1.0, 15.0, 8.0)
    AveBedrms = st.number_input("Average Bedrooms (0.1–5)", 0.1, 5.0, 3.0)
    population = st.number_input("Population (1–10,000)", 1.0, 10000.0, 2000.0)
    AveOccup = st.number_input("Average Occupancy (0.1–20)", 0.1, 20.0, 3.0)

    selected_zip_display = st.selectbox(
        "Select ZIP Code",
        zip_df["zip_display"].tolist()
    )
    selected_row = zip_df[zip_df["zip_display"] == selected_zip_display].iloc[0]
    latitude = selected_row["latitude"]
    longitude = selected_row["longitude"]

    median_income_usd = st.number_input(
        "Median Income ($)", min_value=10000, max_value=200000, value=60000, step=1000
    )
    median_income = median_income_usd / 10_000

    # ------------------------------------------------------------
    # Predict button
    # ------------------------------------------------------------
    if st.button("Predict House Value"):
        input_df = pd.DataFrame([[ 
            longitude,
            latitude,
            housing_median_age,
            population,
            median_income,
            AveRooms,
            AveBedrms,
            AveOccup
        ]], columns=[
            "longitude",
            "latitude",
            "housing_median_age",
            "population",
            "median_income",
            "AveRooms",
            "AveBedrms",
            "AveOccup"
        ])

        predicted_price = model.predict(input_df)[0]

        # Store in session_state
        st.session_state["prediction_made"] = True
        st.session_state["predicted_price"] = predicted_price
        st.session_state["user_input"] = input_df

        st.success(f"🏡 Predicted Median House Value: **${predicted_price:,.2f}**")

        # ------------------------------------------------------------
        # Confidence band & risk label visualization
        # ------------------------------------------------------------
        artifacts = load_error_artifacts()
        y_true = artifacts["y_true"]
        residuals = artifacts["residuals"]

        price_bins = pd.qcut(y_true, q=5)
        df_errors = pd.DataFrame({
            "Actual Price": y_true,
            "Residual": residuals,
            "Price Bin": price_bins
        })

        user_bin = df_errors.loc[
            df_errors["Actual Price"]
            .sub(predicted_price)
            .abs()
            .idxmin(),
            "Price Bin"
        ]

        bin_errors = df_errors[df_errors["Price Bin"] == user_bin]["Residual"]

        lower = bin_errors.quantile(0.1)
        upper = bin_errors.quantile(0.9)
        band_width = upper - lower

        # Risk labeling
        if band_width < 50_000:
            risk_label = "🟢 Low uncertainty"
        elif 50_000 <= band_width < 100_000:
            risk_label = "🟡 Moderate uncertainty"
        else:
            risk_label = "🔴 High uncertainty"

        st.metric(
            label="Prediction Risk Level",
            value=risk_label,
            help=(
                "This risk level is based on historical prediction error "
                "for homes in a similar price range."
            )
        )

        st.caption(
            f"Expected error range (historical): "
            f"**${lower:,.0f} to ${upper:,.0f}**"
        )

        fig, ax = plt.subplots()

        ax.scatter(y_true, residuals, alpha=0.25)
        ax.axvline(predicted_price, linestyle="--", label="Your Prediction")
        ax.axhline(0)

        ax.fill_betweenx(
            [lower, upper],
            predicted_price * 0.98,
            predicted_price * 1.02,
            alpha=0.3,
            label="Local error band"
        )

        ax.set_xlabel("Actual Home Value")
        ax.set_ylabel("Residual (Actual − Predicted)")
        ax.set_title("Historical Error Context at Your Price Level")
        ax.legend()
        st.pyplot(fig)

        st.info(
            "This confidence band reflects historical model behavior for homes "
            "at similar price levels. It represents expected variability—not a guarantee."
        )

    # ============================================================
    # ERROR ANALYSIS (DISPLAY ONLY AFTER PREDICTION)
    # ============================================================
    if st.session_state.get("prediction_made", False):

        st.divider()
        st.title("📊 Prediction Error Context & Model Behavior")
        st.caption("Your prediction compared against historical model error patterns")

        artifacts = load_error_artifacts()

        X_test = artifacts["X_test"]
        y_true = artifacts["y_true"]
        y_pred = artifacts["y_pred"]
        residuals = artifacts["residuals"]
        metrics = artifacts["metrics"]

        # -------------------------------
        # Metrics
        # -------------------------------
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"${metrics['mae']:,.0f}")
        col2.metric("RMSE", f"${metrics['rmse']:,.0f}")
        col3.metric("R²", f"{metrics['r2']:.3f}")

        # -------------------------------
        # Contextualize user's prediction
        # -------------------------------
        user_price = st.session_state["predicted_price"]

        high_price_threshold = np.quantile(y_true, 0.66)

        if user_price >= high_price_threshold:
            st.warning(
                "⚠️ Your predicted home value falls within the **high-value segment**, "
                "where the model historically shows larger and more frequent underpredictions."
            )
        else:
            st.info(
                "ℹ️ Your predicted home value falls within a price range where model errors "
                "tend to be smaller and more stable."
            )

        st.divider()

        # -------------------------------
        # Tabs
        # -------------------------------
        tab1, tab2, tab3 = st.tabs([
            "📊 Price Segment Errors",
            "🌍 Geographic Bias",
            "⚠️ High-Risk Predictions"
        ])

        # ============================================================
        # Tab 1
        # ============================================================
        with tab1:
            st.subheader("How prediction errors vary by price segment")

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
                "Conclusion: Prediction errors are larger and more negative for expensive homes. "
                "The model systematically underpredicts premium properties, increasing financial risk."
            )

        # ============================================================
        # Tab 2
        # ============================================================
        with tab2:
            st.subheader("Geographic distribution of prediction errors")

            fig, ax = plt.subplots()
            scatter = ax.scatter(
                X_test["longitude"],
                X_test["latitude"],
                c=residuals,
                alpha=0.6,
                cmap="coolwarm"
            )

            # Overlay user location
            selected_row = st.session_state["user_input"].iloc[0]
            ax.scatter(
                selected_row["longitude"],
                selected_row["latitude"],
                color="black",
                s=100,
                marker="x",
                label="Your Prediction"
            )
            ax.legend()

            plt.colorbar(scatter, ax=ax, label="Residual")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Geographic Error Patterns")
            st.pyplot(fig)

            st.info(
                "Conclusion: Error magnitude varies by location, with coastal and dense urban regions "
                "showing higher variance. Your prediction is plotted for geographic context."
            )

        # ============================================================
        # Tab 3
        # ============================================================
        with tab3:
            st.subheader("Largest historical prediction errors")

            df_risk = X_test.copy()
            df_risk["Actual Price"] = y_true
            df_risk["Predicted Price"] = y_pred
            df_risk["Residual"] = residuals
            df_risk["Absolute Error"] = np.abs(residuals)

            top_n = st.slider("Show top N highest-risk predictions", 5, 50, 10)

            worst_cases = df_risk.sort_values(
                "Absolute Error", ascending=False
            ).head(top_n)

            st.dataframe(
                worst_cases[
                    ["Actual Price", "Predicted Price", "Residual",
                     "Absolute Error", "latitude", "longitude"]
                ],
                use_container_width=True
            )

            st.warning(
                "Conclusion: Extreme underpredictions represent the highest potential financial harm "
                "and require careful interpretation in real-world use."
            )

        st.divider()
        st.caption(
            "Predictions are estimates. Error analysis is based on historical test data "
            "and provides context—not guarantees—for individual predictions."
        )


# ------------------------------------------------------------
# Run standalone
# ------------------------------------------------------------
if __name__ == "__main__":
    run()
