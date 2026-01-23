import streamlit as st
import pickle
import pandas as pd
import os
import subprocess

# Train models automatically on cloud
if not os.path.exists("model/saved_models.pkl"):
    subprocess.run(["python", "model/train_models.py"])

# Load models
with open("model/saved_models.pkl", "rb") as f:
    models = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Upload dataset
uploaded_file = st.file_uploader("Upload Test CSV (with Activity column)", type=["csv"])

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Activity" not in df.columns:
        st.error("❌ Target column 'Activity' not found in CSV")
    else:
        X = df.drop("Activity", axis=1)
        y = df["Activity"]

        # Encode true labels
        y_encoded = label_encoder.transform(y)

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict (encoded)
        model = models[model_name]
        y_pred_encoded = model.predict(X_scaled)

        # Decode predictions (for display only)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        st.subheader("📊 Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y_encoded, y_pred_encoded))
        st.write("Precision:", precision_score(y_encoded, y_pred_encoded, average="weighted"))
        st.write("Recall:", recall_score(y_encoded, y_pred_encoded, average="weighted"))
        st.write("F1 Score:", f1_score(y_encoded, y_pred_encoded, average="weighted"))

        st.subheader("📌 Confusion Matrix (Encoded)")
        st.write(confusion_matrix(y_encoded, y_pred_encoded))

        st.subheader("📄 Classification Report")
        st.text(
            classification_report(
                y_encoded,
                y_pred_encoded,
                target_names=label_encoder.classes_
            )
        )




