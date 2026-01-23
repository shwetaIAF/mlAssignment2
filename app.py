import os
import pickle
import subprocess
import streamlit as st
import pandas as pd

st.title("ML Assignment 2 - Model Comparison")

# If models not present, train them
if not os.path.exists("model/saved_models.pkl"):
    st.info("Training models for first time... please wait ⏳")
    subprocess.run(["python", "model/train_models.py"])

# Load models
with open("model/saved_models.pkl", "rb") as f:
    models = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.success("Models loaded successfully ✅")

st.write("Upload test.csv to evaluate models")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = df.drop("Activity", axis=1)

    X_scaled = scaler.transform(X)

    results = {}
    for name, model in models.items():
        preds = model.predict(X_scaled)
        results[name] = preds[:5]

    st.write(results)
