
import os
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Create model folder if not exists
os.makedirs("model", exist_ok=True)

print("Downloading dataset from UCI...")

# UCI HAR Dataset (official)
url = "https://raw.githubusercontent.com/selva86/datasets/master/HARtrain.csv"
df = pd.read_csv(url)

# Target and features
X = df.drop("Activity", axis=1)
y = df["Activity"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training models...")

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
    print(f"{name} trained")

with open("model/saved_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

print("✅ All models trained and saved successfully!")
