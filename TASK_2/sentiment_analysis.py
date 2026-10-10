"""
TASK 2: Sentiment Analysis on Text Data

"""

import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Keep the run fast and reproducible: sample a manageable, stratified subset
TRAIN_SAMPLE_SIZE = 6000
TEST_SAMPLE_SIZE = 1500


# 1. LOAD DATA
print("=" * 60)
print("STEP 1: LOADING DATASET (TweetEval - Sentiment)")
print("=" * 60)

def load_split(text_file, label_file):
    with open(text_file, encoding="utf-8") as f:
        texts = [line.strip() for line in f]
    with open(label_file, encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f]
    return pd.DataFrame({"text": texts, "label": labels})

train_df_full = load_split("train_text.txt", "train_labels.txt")
test_df_full = load_split("test_text.txt", "test_labels.txt")
train_df_full["sentiment"] = train_df_full["label"].map(LABEL_MAP)
test_df_full["sentiment"] = test_df_full["label"].map(LABEL_MAP)

print(f"Full train set: {len(train_df_full)} tweets | Full test set: {len(test_df_full)} tweets")

# Stratified sub-sample for a fast, manageable run
def stratified_sample(df, per_class_n):
    parts = [grp.sample(min(len(grp), per_class_n), random_state=RANDOM_STATE)
             for _, grp in df.groupby("sentiment")]
    return pd.concat(parts).reset_index(drop=True)

train_df = stratified_sample(train_df_full, TRAIN_SAMPLE_SIZE // 3)
test_df = stratified_sample(test_df_full, TEST_SAMPLE_SIZE // 3)

print(f"\nUsing sampled train: {len(train_df)} tweets | sampled test: {len(test_df)} tweets")
print(f"\nTrain class balance:\n{train_df['sentiment'].value_counts()}")
print(f"\nSample tweets:")
print(train_df[["text", "sentiment"]].head(3).to_string(index=False))


# 2. PREPROCESS TEXT: tokenization + stopword removal
print("\n" + "=" * 60)
print("STEP 2: TEXT PREPROCESSING")
print("=" * 60)

stop_words = set(stopwords.words("english"))

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                 # remove @mentions
    text = re.sub(r"[^a-z\s]", " ", text)             # keep letters only
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens

train_df["tokens"] = train_df["text"].apply(clean_and_tokenize)
test_df["tokens"] = test_df["text"].apply(clean_and_tokenize)
train_df["clean_text"] = train_df["tokens"].apply(lambda toks: " ".join(toks))
test_df["clean_text"] = test_df["tokens"].apply(lambda toks: " ".join(toks))

print("Example before/after cleaning:")
for i in range(2):
    print(f"  RAW  : {train_df['text'].iloc[i]}")
    print(f"  CLEAN: {train_df['clean_text'].iloc[i]}\n")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])
y_train = train_df["label"].values
y_test = test_df["label"].values

print(f"TF-IDF matrix: train {X_train.shape}, test {X_test.shape}")


# 3. TRAIN MODELS
print("\n" + "=" * 60)
print("STEP 3: TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Naive Bayes": MultinomialNB(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {"model": model, "y_pred": y_pred, "accuracy": acc,
                      "precision": prec, "recall": rec, "f1": f1, "cm": cm}

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]
print(f"Best model: {best_name} (accuracy={best['accuracy']:.4f})")

# 4. PLOTS
plt.figure(figsize=(6, 5))
sns.heatmap(best["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"])
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
plt.ylim(0, 1.0)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# Class distribution plot
plt.figure(figsize=(6, 4))
train_df["sentiment"].value_counts().reindex(["negative", "neutral", "positive"]).plot(kind="bar", color=["#d62728", "#7f7f7f", "#2ca02c"])
plt.title("Training Data Class Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close()


# 5. SAVE RESULTS

with open("results_summary.txt", "w") as f:
    f.write("TASK 2: Sentiment Analysis on Text Data\n")
    f.write("Dataset: TweetEval Sentiment (Cardiff NLP) - github.com/cardiffnlp/tweeteval\n")
    f.write(f"Train samples used: {len(train_df)} | Test samples used: {len(test_df)}\n\n")
    for name, r in results.items():
        f.write(f"--- {name} ---\n")
        f.write(f"Accuracy : {r['accuracy']:.4f}\n")
        f.write(f"Precision: {r['precision']:.4f}\n")
        f.write(f"Recall   : {r['recall']:.4f}\n")
        f.write(f"F1-score : {r['f1']:.4f}\n\n")
    f.write(f"Best model: {best_name}\n")

print("\nAll plots and results saved successfully.")
