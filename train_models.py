import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/train.csv")

X = df.drop("Activity", axis=1)
y = df["Activity"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Define models
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
    )
}

trained_models = {}

# Train models
for name, model in models.items():
    model.fit(X_scaled, y_encoded)
    trained_models[name] = model

# Save models
with open("model/saved_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

print("✅ All models trained successfully")
