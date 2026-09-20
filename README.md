# AI & Machine Learning Task Submission

Two end-to-end machine learning mini-projects, covering the full workflow from raw data to
evaluated model: dataset selection, preprocessing, training, evaluation, and reporting.

| Task | Type | Dataset | Best Model | Accuracy |
|---|---|---|---|---|
| [Task 1](#task-1-classification-model-on-a-standard-dataset) | Classification | Iris Species (Kaggle) | Logistic Regression | 93.3% |
| [Task 2](#task-2-sentiment-analysis-on-text-data) | Sentiment Analysis (NLP) | TweetEval Sentiment (Cardiff NLP) | Logistic Regression | 51.7% |

---

## Task 1: Classification Model on a Standard Dataset

Trains and evaluates a classifier to predict flower species from measurements.

**Dataset:** [Iris Species](https://www.kaggle.com/datasets/uciml/iris) (Kaggle) — 150 samples,
3 species, 4 numeric features.

**Steps:**
1. Load `Iris_kaggle.csv`
2. Preprocess: 80/20 stratified train/test split + `StandardScaler`
3. Train Logistic Regression and Random Forest (Scikit-learn)
4. Evaluate: accuracy, precision, recall, F1, confusion matrix

**Results:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.933** | 0.933 | 0.933 | 0.933 |
| Random Forest | 0.900 | 0.902 | 0.900 | 0.900 |

Petal length and petal width are by far the most predictive features. *Setosa* is perfectly
separable; most confusion happens between *versicolor* and *virginica*.

**Files (in `task1_classification/`):** `classify_iris_final.py`, `Iris_kaggle.csv`, `confusion_matrix.png`,
`model_comparison.png`, `feature_importance.png`, `results_summary.txt`

**Run it:**
```bash
cd task1_classification
pip install scikit-learn pandas numpy matplotlib seaborn
python classify_iris_final.py
```

---

## Task 2: Sentiment Analysis on Text Data

Trains and evaluates a classifier to predict tweet sentiment (negative / neutral / positive).

**Dataset:** [TweetEval — Sentiment](https://github.com/cardiffnlp/tweeteval) (Cardiff NLP) —
real tweets with 3-class sentiment labels. A stratified sample of 6,000 training tweets and
1,500 test tweets (balanced across classes) was used.

**Steps:**
1. Load `train_text.txt` / `train_labels.txt` and `test_text.txt` / `test_labels.txt`
2. Preprocess: clean text (strip URLs/mentions), tokenize (NLTK), remove stopwords (NLTK),
   vectorize with TF-IDF (unigrams + bigrams)
3. Train Logistic Regression and Naive Bayes (Scikit-learn)
4. Evaluate: accuracy, precision, recall, F1, confusion matrix

**Results:**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.517** | 0.534 | 0.517 | 0.512 |
| Naive Bayes | 0.511 | 0.517 | 0.511 | 0.496 |

**Limitation:** TF-IDF is a bag-of-words representation, so it can't capture context, sarcasm,
or negation scope ("not bad" vs "bad"). This shows up clearly in the results — the model
struggles most to identify the **neutral** class (~0.4 precision/recall), which depends on
subtle tone rather than obvious sentiment words. A context-aware model (e.g. a fine-tuned
transformer like BERT) would likely do noticeably better.

**Files (in `task2_sentiment/`):** `sentiment_analysis.py`, `train_text.txt`, `train_labels.txt`, `test_text.txt`,
`test_labels.txt`, `mapping.txt`, `confusion_matrix.png`, `model_comparison.png`,
`class_distribution.png`, `results_summary.txt`

**Run it:**
```bash
cd task2_sentiment
pip install scikit-learn pandas numpy matplotlib seaborn nltk
python sentiment_analysis.py
```

---

## Repository Structure

```
your-repo/
├── README.md
├── task1_classification/
│   ├── classify_iris_final.py
│   ├── Iris_kaggle.csv
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── results_summary.txt
└── task2_sentiment/
    ├── sentiment_analysis.py
    ├── train_text.txt
    ├── train_labels.txt
    ├── test_text.txt
    ├── test_labels.txt
    ├── mapping.txt
    ├── confusion_matrix.png
    ├── model_comparison.png
    ├── class_distribution.png
    └── results_summary.txt
```

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
```

## Author

Submitted as part of an AI & Machine Learning task list assignment.
