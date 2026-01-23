import os
import subprocess
import pickle
import streamlit as st
import pandas as pd

st.title("ML Assignment 2 - Model Comparison")

# Absolute base path (VERY IMPORTANT for Streamlit)
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILE = os.path.join(MODEL_DIR, "saved_models.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")

# STEP 1: Train models only if not present
if not os.path.exists(MODEL_FILE):
    st.info("Training models for first time... please wait ⏳ (about 40 seconds)")
    subprocess.run(["python", os.path.join(MODEL_DIR, "train_models.py")])

# STEP 2: Load models
with open(MODEL_FILE, "rb") as f:
    models = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_FILE, "rb") as f:
    label_encoder = pickle.load(f)

st.success("Models loaded successfully ✅")

# STEP 3: Upload test file
st.write("### Upload test.csv to evaluate models")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Activity" not in df.columns:
        st.error("CSV must contain 'Activity' column")
    else:
        X = df.drop("Activity", axis=1)
        X_scaled = scaler.transform(X)

        st.write("### Sample Predictions (first 5 rows)")

        results = {}
        for name, model in models.items():
            preds = model.predict(X_scaled)
            preds_labels = label_encoder.inverse_transform(preds)
            results[name] = preds_labels[:5]

        st.write(pd.DataFrame(results))
