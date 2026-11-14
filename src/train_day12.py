"""Day 12: a classical sentiment baseline for product reviews (TF-IDF + logistic regression).

Before using a big language model on text, build the cheap classical model first. It is fast, easy to explain,
and it is the number a fancier model (Day 13: BERT) has to beat.

Steps:
    1. Load a balanced sample of Amazon reviews and clean them (HTML, web addresses, extra spaces).
    2. Turn each review into numbers with TF-IDF (word and word-pair counts, weighted by how rare a word is).
    3. Train a logistic regression on the training reviews and score it on reviews it has never seen.
    4. Look at which words push a review toward positive or negative, and save the model.

Run from the main project folder (about 1 minute; the first run downloads a 117 MB review file):
    python -m src.train_day12

Needs:   pip install huggingface_hub pyarrow
"""
import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.config import DAY12_RESULTS_FILE, MODEL_FOLDER, RANDOM_STATE, SENTIMENT_TFIDF_FILE
from src.review_data import SENTIMENTS, load_reviews

TRAIN_PER_CLASS = 2000
TEST_PER_CLASS = 500


def build_model():
    """TF-IDF features, then logistic regression, joined in one pipeline so the same steps run when we predict."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
        ("classifier", LogisticRegression(max_iter=1000, C=4.0, random_state=RANDOM_STATE)),
    ])


def score(y_true, y_guess):
    """Accuracy plus precision, recall and F1 for the 'positive' class, and the confusion matrix."""
    matrix = confusion_matrix(y_true, y_guess)
    return {"accuracy": round(float(accuracy_score(y_true, y_guess)), 4),
            "precision": round(float(precision_score(y_true, y_guess)), 4),
            "recall": round(float(recall_score(y_true, y_guess)), 4),
            "f1": round(float(f1_score(y_true, y_guess)), 4),
            "confusion_matrix": {"negative_right": int(matrix[0, 0]), "negative_called_positive": int(matrix[0, 1]),
                                 "positive_called_negative": int(matrix[1, 0]), "positive_right": int(matrix[1, 1])}}


def strongest_words(model, how_many=10):
    """The words with the biggest weights: positive ones push toward 'positive', negative ones toward 'negative'."""
    words = np.array(model.named_steps["tfidf"].get_feature_names_out())
    weights = model.named_steps["classifier"].coef_[0]
    order = np.argsort(weights)
    return {"negative": [str(w) for w in words[order[:how_many]]], "positive": [str(w) for w in words[order[::-1][:how_many]]]}


def main():
    print("Step 1: load and clean the reviews")
    x_train, x_test, y_train, y_test = load_reviews(TRAIN_PER_CLASS, TEST_PER_CLASS)
    print(f"  {len(x_train)} training reviews, {len(x_test)} test reviews (half positive, half negative)")
    print("  example:", x_train[0][:110], "->", SENTIMENTS[int(y_train[0])])

    print("\nStep 2 and 3: TF-IDF + logistic regression")
    model = build_model()
    model.fit(x_train, y_train)
    vocabulary = len(model.named_steps["tfidf"].get_feature_names_out())
    scores = score(y_test, model.predict(x_test))
    train_accuracy = round(float(accuracy_score(y_train, model.predict(x_train))), 4)
    print(f"  {vocabulary} word and word-pair features")
    print(f"  accuracy on reviews it has never seen: {scores['accuracy']} (on its own training reviews: {train_accuracy})")
    print(f"  precision {scores['precision']}  recall {scores['recall']}  F1 {scores['f1']}")

    print("\nStep 4: which words matter?")
    words = strongest_words(model)
    print("  pushes toward negative:", ", ".join(words["negative"]))
    print("  pushes toward positive:", ", ".join(words["positive"]))

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(model, SENTIMENT_TFIDF_FILE)
    with open(DAY12_RESULTS_FILE, "w") as f:
        json.dump({"data": "Amazon reviews (fancyzhx/amazon_polarity test file), balanced sample",
                   "train_reviews": len(x_train), "test_reviews": len(x_test), "features": vocabulary,
                   "train_accuracy": train_accuracy, "test_scores": scores, "strongest_words": words}, f, indent=2)
    print("\nSaved ->", SENTIMENT_TFIDF_FILE, "and", DAY12_RESULTS_FILE)


if __name__ == "__main__":
    main()
