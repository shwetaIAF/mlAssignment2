import os
import pickle
import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.title("ML Assignment 2 - Model Comparison")

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "saved_models.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")

# -------- TRAIN IF NOT EXISTS --------
if not os.path.exists(MODEL_FILE):

    st.info("First run: Training models... please wait ⏳ (~40 sec)")

    url = "train.csv"
    df = pd.read_csv(url)

    X = df.drop("Activity", axis=1)
    y = df["Activity"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(label_encoder, f)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(
            objective="multi:softmax",
            num_class=len(set(y_encoded)),
            eval_metric="mlogloss"
        ),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_scaled, y_encoded)
        trained_models[name] = model

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(trained_models, f)

    st.success("Training completed and models saved ✅")

# -------- LOAD MODELS --------
with open(MODEL_FILE, "rb") as f:
    models = pickle.load(f)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_FILE, "rb") as f:
    label_encoder = pickle.load(f)

st.success("Models loaded successfully ✅")

# -------- TEST FILE UPLOAD --------
st.write("### Upload test.csv to evaluate models")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Activity" not in df.columns:
        st.error("CSV must contain 'Activity' column")
    else:
        X = df.drop("Activity", axis=1)
        X_scaled = scaler.transform(X)

        results = {}
        for name, model in models.items():
            preds = model.predict(X_scaled)
            preds_labels = label_encoder.inverse_transform(preds)
            results[name] = preds_labels[:5]

        st.write(pd.DataFrame(results))

