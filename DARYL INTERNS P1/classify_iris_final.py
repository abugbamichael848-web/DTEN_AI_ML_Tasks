"""
TASK 1: Classification Model on a Standard Dataset
Dataset : Iris Species dataset from Kaggle (https://www.kaggle.com/datasets/uciml/iris)
Library : Scikit-learn

Steps:
1. Load the dataset downloaded from Kaggle (Iris.csv)
2. Preprocess the data (train/test split + feature scaling)
3. Train classification model(s)
4. Evaluate performance: accuracy, precision, recall, confusion matrix
5. Save plots + results summary

HOW TO GET THE DATA:
1. Go to https://www.kaggle.com/datasets/uciml/iris
2. Click "Download" (downloads archive.zip)
3. Unzip it -> Iris.csv
Place Iris.csv in the same folder as this script (see MAIN_DATA_FILE below).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
KAGGLE_DATASET = "uciml/iris"  # https://www.kaggle.com/datasets/uciml/iris
MAIN_DATA_FILE = "Iris_kaggle.csv"  # Iris.csv downloaded from Kaggle

# ---------------------------------------------------------------
# 1. LOAD DATA (downloaded from Kaggle)
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING DATASET DOWNLOADED FROM KAGGLE")
print("=" * 60)
print(f"Kaggle dataset: {KAGGLE_DATASET}")
if not os.path.exists(MAIN_DATA_FILE):
    raise FileNotFoundError(
        f"{MAIN_DATA_FILE} not found. Download it from "
        f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}, unzip, and place Iris.csv "
        f"in this folder as {MAIN_DATA_FILE}."
    )

df = pd.read_csv(MAIN_DATA_FILE)
print(f"Dataset shape: {df.shape}")
print(df.head())
print(f"\nClass balance:\n{df['Species'].value_counts()}\n")

# ---------------------------------------------------------------
# 2. PREPROCESS: train/test split + scaling
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 2: PREPROCESSING")
print("=" * 60)

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols].values

le = LabelEncoder()
y = le.fit_transform(df["Species"])
target_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples\n")

# ---------------------------------------------------------------
# 3. TRAIN MODELS
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 3: TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {"model": model, "y_pred": y_pred, "accuracy": acc,
                      "precision": prec, "recall": rec, "f1": f1, "cm": cm}

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]
print(f"Best model: {best_name} (accuracy={best['accuracy']:.4f})")

# ---------------------------------------------------------------
# 4. EVALUATE + PLOTS
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: EVALUATION PLOTS")
print("=" * 60)

plt.figure(figsize=(6, 5))
sns.heatmap(best["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

metrics_df = pd.DataFrame({
    name: [r["accuracy"], r["precision"], r["recall"], r["f1"]]
    for name, r in results.items()
}, index=["Accuracy", "Precision", "Recall", "F1"])
metrics_df.plot(kind="bar", figsize=(7, 5), rot=0)
plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0.8, 1.02)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(6, 4))
importances.plot(kind="barh", color="teal")
plt.title("Random Forest — Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. SAVE RESULTS SUMMARY
# ---------------------------------------------------------------
with open("results_summary.txt", "w") as f:
    f.write("TASK 1: Classification Model on a Standard Dataset\n")
    f.write(f"Dataset source: Kaggle - '{KAGGLE_DATASET}' (Iris.csv)\n\n")
    for name, r in results.items():
        f.write(f"--- {name} ---\n")
        f.write(f"Accuracy : {r['accuracy']:.4f}\n")
        f.write(f"Precision: {r['precision']:.4f}\n")
        f.write(f"Recall   : {r['recall']:.4f}\n")
        f.write(f"F1-score : {r['f1']:.4f}\n\n")
    f.write(f"Best model: {best_name}\n")

print("\nAll plots and results saved successfully.")
