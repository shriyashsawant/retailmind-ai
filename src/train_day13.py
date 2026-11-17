"""Day 13: use BERT for review sentiment and support-ticket routing, and check if it beats the Day 12 baseline.

BERT is a large language model that has already read a huge amount of text. We use a small version of it,
DistilBERT (about 66 million weights, 270 MB), in two ways:
    - as a FEATURE EXTRACTOR: BERT is frozen; it turns each review into 768 numbers; a logistic regression
      (the same simple classifier as Day 12) learns from those numbers.
    - FINE-TUNED: we also let BERT's own weights change a little while it learns the task (1 pass over the reviews).

Steps:
    1. Load the same reviews and the same train/test split as Day 12.
    2. Sentiment with three models: TF-IDF (Day 12), frozen DistilBERT, fine-tuned DistilBERT.
    3. Is BERT's lift real? A paired bootstrap on the test reviews gives a range for each difference.
       And what does it cost? Speed and file size.
    4. Support tickets: relabel reviews as tickets (shipping, quality, refund, how-to) and route them with
       TF-IDF and with frozen DistilBERT; compare per category.
    5. Save the BERT-based classifiers for the pipeline (src/sentiment_service.py).

Run from the main project folder (about 25 to 35 minutes on a laptop CPU; downloads DistilBERT, 270 MB, once):
    python -m src.train_day13          (set QUICK=1 for a 1-minute smoke test)

Needs:   pip install torch transformers huggingface_hub pyarrow
"""
import json
import os
import time

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.config import DAY13_RESULTS_FILE, MODEL_FOLDER, RANDOM_STATE, SENTIMENT_BERT_FILE, SENTIMENT_TFIDF_FILE
from src.review_data import SENTIMENTS, clean_text, load_reviews, make_ticket_data

MODEL_NAME = "distilbert/distilbert-base-uncased"
TICKET_HEAD_FILE = "models/ticket_router.joblib"
QUICK = bool(os.environ.get("QUICK"))
TRAIN_PER_CLASS, TEST_PER_CLASS = (30, 15) if QUICK else (2000, 500)
MAX_LENGTH = 128
BATCH = 32


def embed(texts, tokenizer, model):
    """Turn each text into 768 numbers: the average of BERT's output for every word (padding is ignored)."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = tokenizer(texts[start:start + BATCH], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            hidden = model(**batch).last_hidden_state                          # (reviews, words, 768)
            mask = batch["attention_mask"].unsqueeze(-1).float()               # 1 for real words, 0 for padding
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
    return np.vstack(out)


def fine_tune(tokenizer, x_train, y_train):
    """One pass over the training reviews, letting all of BERT's weights change a little (learning rate 5e-5)."""
    torch.manual_seed(RANDOM_STATE)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    order = np.random.RandomState(RANDOM_STATE).permutation(len(x_train))
    model.train()
    for step, start in enumerate(range(0, len(order), 16)):
        chosen = order[start:start + 16]
        batch = tokenizer([x_train[i] for i in chosen], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        loss = model(**batch, labels=torch.tensor(y_train[chosen], dtype=torch.long)).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 25 == 0:
            print(f"    step {step} of {len(order) // 16}, loss {loss.item():.3f}", flush=True)
    return model


def predict_fine_tuned(model, tokenizer, texts):
    model.eval()
    guesses = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = tokenizer(texts[start:start + BATCH], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            guesses.append(model(**batch).logits.argmax(-1).numpy())
    return np.concatenate(guesses)


def paired_bootstrap(correct_a, correct_b, rounds=2000):
    """How much better is model B than model A? Resample the test reviews 2,000 times; return the 95% range
    of (accuracy B minus accuracy A). If the range includes 0, the difference could be luck."""
    random = np.random.RandomState(RANDOM_STATE)
    difference = correct_b.astype(float) - correct_a.astype(float)
    means = [difference[random.randint(0, len(difference), len(difference))].mean() for _ in range(rounds)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def per_class_recall(y_true, y_guess, classes):
    return {str(c): round(float(np.mean(y_guess[y_true == c] == c)), 3) for c in classes}


def bert_head():
    """The small classifier that sits on top of BERT's 768 numbers."""
    return Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, C=0.05, random_state=RANDOM_STATE))])


def main():
    print("Step 1: the same reviews and split as Day 12")
    x_train, x_test, y_train, y_test = load_reviews(TRAIN_PER_CLASS, TEST_PER_CLASS)
    print(f"  {len(x_train)} training reviews, {len(x_test)} test reviews" + ("  (QUICK smoke test)" if QUICK else ""))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)

    print("\nStep 2: sentiment with three models")
    tfidf = joblib.load(SENTIMENT_TFIDF_FILE)                                    # the Day 12 model
    started = time.time()
    tfidf_guess = tfidf.predict(x_test)
    tfidf_seconds = time.time() - started
    print(f"  TF-IDF (Day 12):          accuracy {accuracy_score(y_test, tfidf_guess):.4f}")

    started = time.time()
    train_features = embed(x_train, tokenizer, backbone)
    print(f"  embedded {len(x_train)} training reviews in {time.time() - started:.0f} s", flush=True)
    started = time.time()
    test_features = embed(x_test, tokenizer, backbone)
    bert_seconds = time.time() - started
    head = bert_head().fit(train_features, y_train)
    frozen_guess = head.predict(test_features)
    print(f"  frozen DistilBERT + LR:   accuracy {accuracy_score(y_test, frozen_guess):.4f}", flush=True)

    print("  fine-tuning DistilBERT (one pass; this is the slow part)")
    tuned = fine_tune(tokenizer, x_train, y_train)
    tuned_guess = predict_fine_tuned(tuned, tokenizer, x_test)
    print(f"  fine-tuned DistilBERT:    accuracy {accuracy_score(y_test, tuned_guess):.4f}", flush=True)

    guesses = {"TF-IDF + LR (Day 12)": tfidf_guess, "DistilBERT frozen + LR": frozen_guess, "DistilBERT fine-tuned": tuned_guess}
    sentiment = {name: {"accuracy": round(float(accuracy_score(y_test, g)), 4), "f1": round(float(f1_score(y_test, g)), 4)}
                 for name, g in guesses.items()}

    print("\nStep 3: is the lift real, and what does it cost?")
    correct = {name: (g == y_test) for name, g in guesses.items()}
    lift = {}
    for name in ("DistilBERT frozen + LR", "DistilBERT fine-tuned"):
        lift[name] = {"accuracy_minus_tfidf": round(sentiment[name]["accuracy"] - sentiment["TF-IDF + LR (Day 12)"]["accuracy"], 4),
                      "range_95": paired_bootstrap(correct["TF-IDF + LR (Day 12)"], correct[name])}
        print(f"  {name}: {lift[name]['accuracy_minus_tfidf']:+.4f} against TF-IDF, 95% range {lift[name]['range_95']}")
    cost = {"tfidf_reviews_per_second": round(len(x_test) / max(tfidf_seconds, 1e-6)),
            "bert_reviews_per_second": round(len(x_test) / bert_seconds, 1),
            "tfidf_file_mb": round(os.path.getsize(SENTIMENT_TFIDF_FILE) / 1e6, 1), "bert_download_mb": 268}
    print(f"  speed: TF-IDF {cost['tfidf_reviews_per_second']} reviews/s, DistilBERT {cost['bert_reviews_per_second']} reviews/s")
    print(f"  size:  TF-IDF {cost['tfidf_file_mb']} MB, DistilBERT {cost['bert_download_mb']} MB")

    print("\nStep 4: support-ticket routing (tickets are relabelled reviews, trigger words removed)")
    pool, _, _, _ = load_reviews(15000 if not QUICK else 200, 0)              # a larger pool of reviews to find tickets in
    ticket_texts, ticket_labels = make_ticket_data(pool)
    ticket_labels = np.array([str(c) for c in ticket_labels])                  # plain text labels
    categories = sorted(set(ticket_labels))
    counts = {c: int((ticket_labels == c).sum()) for c in categories}
    print("  tickets found per category:", counts)
    cap = min(min(counts.values()), 30 if QUICK else 500)
    random = np.random.RandomState(RANDOM_STATE)
    chosen = np.concatenate([random.choice(np.where(ticket_labels == c)[0], cap, replace=False) for c in categories])
    random.shuffle(chosen)
    t_texts, t_labels = [ticket_texts[i] for i in chosen], ticket_labels[chosen]
    cut = int(len(t_texts) * 0.8)
    tx_train, tx_test, ty_train, ty_test = t_texts[:cut], t_texts[cut:], t_labels[:cut], t_labels[cut:]
    print(f"  balanced to {cap} per category: {len(tx_train)} training tickets, {len(tx_test)} test tickets")

    ticket_tfidf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                             ("classifier", LogisticRegression(max_iter=1000, C=4.0, random_state=RANDOM_STATE))]).fit(tx_train, ty_train)
    ticket_bert = bert_head().fit(embed(tx_train, tokenizer, backbone), ty_train)
    bert_ticket_guess = ticket_bert.predict(embed(tx_test, tokenizer, backbone))
    tfidf_ticket_guess = ticket_tfidf.predict(tx_test)
    tickets = {
        "categories": categories, "per_category": cap, "train": len(tx_train), "test": len(tx_test),
        "TF-IDF + LR": {"accuracy": round(float(accuracy_score(ty_test, tfidf_ticket_guess)), 4),
                        "recall_per_category": per_class_recall(ty_test, tfidf_ticket_guess, categories)},
        "DistilBERT frozen + LR": {"accuracy": round(float(accuracy_score(ty_test, bert_ticket_guess)), 4),
                                   "recall_per_category": per_class_recall(ty_test, bert_ticket_guess, categories)},
        "difference_range_95": paired_bootstrap(tfidf_ticket_guess == ty_test, bert_ticket_guess == ty_test),
    }
    for name in ("TF-IDF + LR", "DistilBERT frozen + LR"):
        print(f"  {name:<24} accuracy {tickets[name]['accuracy']}  per category {tickets[name]['recall_per_category']}")
    print(f"  BERT minus TF-IDF, 95% range: {tickets['difference_range_95']}")

    print("\nStep 5: save the BERT-based classifiers")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(head, SENTIMENT_BERT_FILE)
    joblib.dump({"categories": categories, "head": ticket_bert}, TICKET_HEAD_FILE)
    tuned.save_pretrained("models/sentiment_distilbert_finetuned")            # large (about 270 MB): kept out of git
    tokenizer.save_pretrained("models/sentiment_distilbert_finetuned")
    with open(DAY13_RESULTS_FILE, "w") as f:
        json.dump({"quick_smoke_test": QUICK, "model": MODEL_NAME, "train_reviews": len(x_train), "test_reviews": len(x_test),
                   "sentiment": sentiment, "lift_over_tfidf": lift, "cost": cost, "tickets": tickets}, f, indent=2)
    print("  Saved ->", SENTIMENT_BERT_FILE, ",", TICKET_HEAD_FILE, "and", DAY13_RESULTS_FILE)


if __name__ == "__main__":
    main()
