import streamlit as st
import pandas as pd
import joblib
import os

st.markdown(
    """
    <style>
    .stApp {
        background-color: lightgray; 
    h1{
       font-size:35px;
       text-align:center;
       color:navy;
    }
    h3{
       margin-bottom:10px;
       font-size:20px;
       text-align:center;
       color:mediumblue;
    }

    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "models/xgb_model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Predictive Maintenance - RUL Estimator")

st.subheader("Upload engine sensor data (CSV) and get RUL predictions")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if "RUL" in data.columns:  
        data = data.drop(columns=["RUL"])

    preds = model.predict(data)

    results = pd.DataFrame({"Predicted_RUL": preds})
    st.write("Predictions:", results.head())

    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predicted_rul.csv",
        mime="text/csv",
    )
