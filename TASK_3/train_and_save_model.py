"""
Trains the Task 1 classification model (Logistic Regression on the Iris dataset)
and saves it + the scaler + label encoder to disk so the Flask API can load them.
"""

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "Iris_kaggle.csv"))
feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols].values

le = LabelEncoder()
y = le.fit_transform(df["Species"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

joblib.dump(model, os.path.join(BASE_DIR, "iris_model.joblib"))
joblib.dump(scaler, os.path.join(BASE_DIR, "iris_scaler.joblib"))
joblib.dump(le, os.path.join(BASE_DIR, "iris_label_encoder.joblib"))

print("Saved: iris_model.joblib, iris_scaler.joblib, iris_label_encoder.joblib")
print(f"Classes: {list(le.classes_)}")
print(f"Feature order expected: {feature_cols}")