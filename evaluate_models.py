import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize

# Load test data
df = pd.read_csv("data/test.csv")

X_test = df.drop("Activity", axis=1)
y_test = df["Activity"]

# Load scaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)

# Load label encoder
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Encode true labels
y_test_encoded = label_encoder.transform(y_test)

# Load trained models
with open("model/saved_models.pkl", "rb") as f:
    models = pickle.load(f)

results = []

# Number of classes
classes = list(range(len(label_encoder.classes_)))

for name, model in models.items():
    # Predict (encoded)
    y_pred_encoded = model.predict(X_test_scaled)

    # Binarize for AUC
    y_test_bin = label_binarize(y_test_encoded, classes=classes)
    y_pred_bin = label_binarize(y_pred_encoded, classes=classes)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_encoded, y_pred_encoded),
        "AUC": roc_auc_score(y_test_bin, y_pred_bin, multi_class="ovr"),
        "Precision": precision_score(y_test_encoded, y_pred_encoded, average="weighted"),
        "Recall": recall_score(y_test_encoded, y_pred_encoded, average="weighted"),
        "F1": f1_score(y_test_encoded, y_pred_encoded, average="weighted"),
        "MCC": matthews_corrcoef(y_test_encoded, y_pred_encoded)
    })

results_df = pd.DataFrame(results)
print(results_df)
