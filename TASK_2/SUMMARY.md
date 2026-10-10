# Task 2: Sentiment Analysis on Text Data

## Dataset
**TweetEval — Sentiment** (Cardiff NLP): https://github.com/cardiffnlp/tweeteval
Real tweets labeled **negative / neutral / positive**, published as part of the TweetEval
NLP benchmark (Barbieri et al., 2020). Full dataset: 45,615 training tweets + 12,284 test
tweets. For a fast, balanced run, a stratified sample of 6,000 training tweets (2,000 per
class) and 1,500 test tweets (500 per class) was used.

## Preprocessing
1. **Cleaning**: lowercased text, stripped URLs and `@mentions`, removed non-letter characters.
2. **Tokenization**: NLTK's `word_tokenize`.
3. **Stopword removal**: NLTK's English stopword list (e.g. "the", "is", "and").
4. **Vectorization**: `TfidfVectorizer` (unigrams + bigrams, top 5,000 features, min document
   frequency 2) — converts cleaned text into numeric TF-IDF vectors for the model.

Example:
> Raw: *"#CNET - Just bought my 1st iPad, iPad3, feeling real burned, mad, about iPad4 so soon..."*
> Cleaned: `cnet bought st ipad ipad feeling real burned mad ipad soon grrr really mad even care mini`

## Model
Two models were trained on the TF-IDF vectors and compared:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Logistic Regression** | **0.517** | 0.534 | 0.517 | 0.512 |
| Naive Bayes | 0.511 | 0.517 | 0.511 | 0.496 |

**Best model: Logistic Regression**, ~52% accuracy on a balanced 3-class test set (33%
random-guess baseline), so the model is clearly learning real signal, just far from perfect.

## Limitation
The **neutral class is the model's biggest weakness** — precision/recall for "neutral" sits
around 0.4, well below "negative" and "positive." This is a known, general limitation of
bag-of-words/TF-IDF approaches to sentiment: TF-IDF treats sentiment as a bag of independent
words and can't capture context, sarcasm, negation scope ("not bad" vs "bad"), or the subtle
tone that separates a mildly-opinionated tweet from a truly neutral one. Real tweets also
contain slang, abbreviations, and emojis that a simple cleaned/tokenized vocabulary doesn't
fully capture. A context-aware model (e.g. a fine-tuned transformer like BERT/RoBERTa) would
likely perform substantially better, especially at distinguishing neutral from mildly positive
or negative text.

## Files in this project
- `train_text.txt` / `train_labels.txt` — training data (downloaded)
- `test_text.txt` / `test_labels.txt` — test data (downloaded)
- `mapping.txt` — label mapping (0=negative, 1=neutral, 2=positive)
- `sentiment_analysis.py` — full pipeline (load → preprocess → vectorize → train → evaluate)
- `confusion_matrix.png` — confusion matrix for the best model
- `model_comparison.png` — accuracy/precision/recall/F1 comparison
- `class_distribution.png` — training data class balance
- `results_summary.txt` — plain-text metrics dump
