# Task 1: Classification Model on a Standard Dataset

## Dataset
**Iris Species** dataset, downloaded from Kaggle: https://www.kaggle.com/datasets/uciml/iris
150 samples, 3 species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*), 4 numeric
features (sepal length/width, petal length/width) plus an `Id` column. Only this single
dataset is used — no other data source or merge step.

## Preprocessing
- Split into 80% train (120 samples) / 20% test (30 samples), stratified by class so each
  species is evenly represented in both sets.
- Applied `StandardScaler` to normalize features to zero mean / unit variance — important
  for Logistic Regression, and harmless for Random Forest.

## Model Choice
Two models were trained and compared using Scikit-learn:

| Model | Why chosen |
|---|---|
| **Logistic Regression** | Simple, fast, interpretable baseline for a linearly-separable-ish problem like Iris. |
| **Random Forest** | Handles non-linear boundaries and gives feature-importance insight, as a stronger comparison model. |

## Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | **0.933** | 0.933 | 0.933 | 0.933 |
| Random Forest | 0.900 | 0.902 | 0.900 | 0.900 |

**Best model: Logistic Regression** (93.3% test accuracy).

- Both models classified *setosa* perfectly (100% precision/recall) — it's linearly separable
  from the other two species.
- Most confusion happens between *versicolor* and *virginica*, which overlap in petal
  measurements — a known characteristic of this dataset.
- Feature importance (from Random Forest) shows **petal length and petal width** are by far
  the most predictive features, far more so than sepal measurements.

## Files in this project
- `Iris_kaggle.csv` — the dataset, downloaded from Kaggle
- `classify_iris_final.py` — full pipeline (load → preprocess → train → evaluate → plot)
- `confusion_matrix.png` — confusion matrix for the best model
- `model_comparison.png` — bar chart comparing both models on all 4 metrics
- `feature_importance.png` — which features mattered most
- `results_summary.txt` — plain-text metrics dump
