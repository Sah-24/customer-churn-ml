import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ------------------------------
# Load Model
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model_pipeline.pkl")
model = joblib.load(MODEL_PATH)

# ------------------------------
# Title & Description
# ------------------------------
st.title("Customer Churn Prediction App")
st.markdown(
    """
    This application predicts whether a telecom customer is likely to **churn**
    based on their demographic details, service usage, and billing information.
    """
)

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Customer Information")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    internet = st.sidebar.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    contract = st.sidebar.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

    payment = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    return pd.DataFrame([data])


input_df = user_input_features()

# ------------------------------
# Prediction
# ------------------------------
st.subheader("Prediction Result")

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("⚠️ High Risk: Customer is likely to churn")
    else:
        st.success("✅ Low Risk: Customer is likely to stay")

    st.metric("Churn Probability", f"{probability:.2%}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Machine Learning Project | Built with Streamlit")
