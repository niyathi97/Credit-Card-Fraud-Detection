import streamlit as st
import pandas as pd
import joblib
import datetime
import time as t

# -----------------------------
# LOAD MODEL & DATA
# -----------------------------
model = joblib.load("models/fraud_model.pkl")
df = pd.read_csv("data/creditcard.csv")

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.markdown("### 🔍 Smart Transaction Analyzer")

st.write("Enter transaction details below:")

# -----------------------------
# USER INPUT (IMPROVED UI)
# -----------------------------
amount = st.slider("💰 Transaction Amount", 0.0, 10000.0, value=500.0, step=100.0)

# Time selection (REALISTIC)
time_mode = st.radio("⏱ Choose Time Mode", ["Current Time", "Custom Time"])

if time_mode == "Current Time":
    now = datetime.datetime.now()
    time_seconds = now.hour * 3600 + now.minute * 60 + now.second
    st.write(f"🕒 Current Time: {now.strftime('%H:%M:%S')}")
else:
    selected_time = st.time_input("Select Transaction Time")
    time_seconds = selected_time.hour * 3600 + selected_time.minute * 60
    st.write(f"🕒 Selected Time: {selected_time}")

# -----------------------------
# BUTTON ACTION
# -----------------------------
if st.button("🚀 Predict Transaction"):
    
    # Take random row from dataset
    sample = df.sample(1).drop("Class", axis=1)

    # Replace key values
    sample["Amount"] = amount
    sample["Time"] = time_seconds

    # -----------------------------
    # MEASURE PREDICTION TIME
    # -----------------------------
    start = t.time()

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    end = t.time()
    latency = (end - start) * 1000  # in milliseconds

    # -----------------------------
    # OUTPUT RESULT
    # -----------------------------
    st.markdown("---")

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Normal Transaction\n\nFraud Probability: {probability:.2f}")

    # -----------------------------
    # SHOW PERFORMANCE
    # -----------------------------
    st.info(f"⏱ Prediction Time: {latency:.2f} ms")

    # -----------------------------
    # SHOW DATA USED
    # -----------------------------
    st.markdown("### 📊 Transaction Details Used")
    st.dataframe(sample)

# -----------------------------
# EXTRA INFO
# -----------------------------
st.markdown("---")
st.info("ℹ️ This system uses Machine Learning to detect fraudulent transactions in real-time.")