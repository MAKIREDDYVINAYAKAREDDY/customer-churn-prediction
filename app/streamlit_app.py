import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_tuned.joblib"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "final_model_config.joblib"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.joblib"
)


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, config, preprocessor


model, config, preprocessor = load_artifacts()

threshold = config["threshold"]


st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Predict customer churn using a tuned XGBoost model
    with SHAP-based explainability.
    """
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


with col2:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )


with col3:

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


st.subheader("Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col2:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=10.0
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True
):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    transformed = preprocessor.transform(customer)

    probability = model.predict_proba(
        transformed
    )[0, 1]

    prediction = int(
        probability >= threshold
    )


    # -----------------------------------------------------
    # Prediction Result
    # -----------------------------------------------------

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )

    with result_col3:

        if prediction == 1:

            st.metric(
                "Prediction",
                "⚠️ CHURN"
            )

        else:

            st.metric(
                "Prediction",
                "✅ NO CHURN"
            )


    # -----------------------------------------------------
    # Risk Level
    # -----------------------------------------------------

    if probability >= 0.70:

        risk = "🔴 High Risk"

    elif probability >= 0.40:

        risk = "🟠 Medium Risk"

    else:

        risk = "🟢 Low Risk"


    st.info(
        f"Customer Risk Level: **{risk}**"
    )

    st.progress(
        float(probability),
        text=f"Churn probability: {probability:.1%}"
    )


    # =====================================================
    # SHAP EXPLAINABILITY
    # =====================================================

    st.divider()

    st.subheader("🔍 Why This Prediction?")

    with st.spinner("Calculating SHAP explanation..."):

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            transformed
        )

        values = shap_values[0]

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        explanation = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": values
        })

        explanation["Impact"] = (
            explanation["SHAP Value"].abs()
        )

        explanation = explanation.sort_values(
            "Impact",
            ascending=False
        )


    # -----------------------------------------------------
    # Top Risk Factors
    # -----------------------------------------------------

    risk_factors = explanation[
        explanation["SHAP Value"] > 0
    ].head(5)

    protective_factors = explanation[
        explanation["SHAP Value"] < 0
    ].head(5)


    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🔴 Factors Increasing Churn"
        )

        if len(risk_factors) == 0:

            st.write(
                "No strong factors increasing churn."
            )

        else:

            for _, row in risk_factors.iterrows():

                feature = row["Feature"]
                value = row["SHAP Value"]

                st.write(
                    f"**{feature}**  "
                    f"`+{value:.4f}`"
                )


    with col2:

        st.markdown(
            "### 🟢 Factors Reducing Churn"
        )

        if len(protective_factors) == 0:

            st.write(
                "No strong protective factors."
            )

        else:

            for _, row in protective_factors.iterrows():

                feature = row["Feature"]
                value = row["SHAP Value"]

                st.write(
                    f"**{feature}**  "
                    f"`{value:.4f}`"
                )


    # -----------------------------------------------------
    # SHAP Bar Chart
    # -----------------------------------------------------

    st.markdown(
        "### Feature Impact"
    )

    chart_data = explanation.head(10).copy()

    chart_data = chart_data[
        ["Feature", "SHAP Value"]
    ]

    chart_data = chart_data.set_index(
        "Feature"
    )

    st.bar_chart(
        chart_data
    )


st.divider()

st.caption(
    "Model: Tuned XGBoost | "
    f"Decision threshold: {threshold:.2f} | "
    "Explainability: SHAP"
)
