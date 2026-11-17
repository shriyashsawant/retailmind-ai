"""The RetailMind text module: review sentiment and support-ticket routing.

    from src.sentiment_service import predict_sentiment, route_ticket

    predict_sentiment(["Love it, works perfectly"])                          # best model that is available
    predict_sentiment(["Love it, works perfectly"], backend="tfidf")         # Day 12 classical model, no big downloads
    predict_sentiment(["Love it, works perfectly"], backend="bert-frozen")   # frozen DistilBERT + small classifier
    predict_sentiment(["Love it, works perfectly"], backend="bert-finetuned")  # fine-tuned DistilBERT
    route_ticket(["The box never showed up and the tracking is stuck"])

Which backend is best (Day 13, 1,000 test reviews): fine-tuned DistilBERT 0.902, TF-IDF 0.870, frozen DistilBERT 0.867.
The default backend="auto" therefore uses the fine-tuned model when its folder exists (python -m src.train_day13 makes it;
it is about 270 MB, so it is not stored in git) and otherwise falls back to TF-IDF.

Run from the main project folder (after src.train_day12 and src.train_day13):
    python -m src.sentiment_service

The BERT backends need:   pip install torch transformers
"""
import os

import joblib

from src.config import SENTIMENT_BERT_FILE, SENTIMENT_TFIDF_FILE
from src.review_data import SENTIMENTS, clean_text

TICKET_HEAD_FILE = "models/ticket_router.joblib"
FINE_TUNED_FOLDER = "models/sentiment_distilbert_finetuned"
_loaded = {}                      # the BERT models are big, so load them once and keep them


def _need_bert():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise ImportError("The BERT backends need torch and transformers:  pip install torch transformers  (or use backend='tfidf').")


def _bert_features(texts):
    """DistilBERT's 768 numbers for each text (the same step used in training)."""
    _need_bert()
    from transformers import AutoModel, AutoTokenizer
    from src.train_day13 import MODEL_NAME, embed
    if "backbone" not in _loaded:
        _loaded["backbone"] = (AutoTokenizer.from_pretrained(MODEL_NAME), AutoModel.from_pretrained(MODEL_NAME))
    tokenizer, backbone = _loaded["backbone"]
    return embed(texts, tokenizer, backbone)


def _fine_tuned_probabilities(texts):
    _need_bert()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if "tuned" not in _loaded:
        _loaded["tuned"] = (AutoTokenizer.from_pretrained(FINE_TUNED_FOLDER),
                            AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_FOLDER).eval())
    tokenizer, model = _loaded["tuned"]
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            batch = tokenizer(texts[start:start + 32], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].numpy())
    return [float(p) for chunk in out for p in chunk]


def predict_sentiment(reviews, backend="auto"):
    """reviews: a list of review texts. Returns one answer per review: the sentiment and the probability it is positive."""
    texts = [clean_text(r) for r in reviews]
    if backend == "auto":
        backend = "bert-finetuned" if os.path.isdir(FINE_TUNED_FOLDER) else "tfidf"
    if backend == "tfidf":
        probabilities = joblib.load(SENTIMENT_TFIDF_FILE).predict_proba(texts)[:, 1]
    elif backend == "bert-frozen":
        probabilities = joblib.load(SENTIMENT_BERT_FILE).predict_proba(_bert_features(texts))[:, 1]
    elif backend == "bert-finetuned":
        if not os.path.isdir(FINE_TUNED_FOLDER):
            raise FileNotFoundError(f"{FINE_TUNED_FOLDER} not found. Run:  python -m src.train_day13")
        probabilities = _fine_tuned_probabilities(texts)
    else:
        raise ValueError("backend must be 'auto', 'tfidf', 'bert-frozen' or 'bert-finetuned'")
    return [{"sentiment": SENTIMENTS[int(p > 0.5)], "probability_positive": round(float(p), 4)} for p in probabilities]


def route_ticket(tickets):
    """tickets: a list of support-ticket texts. Returns the category each one should go to, and the confidence.
    (Day 13 found this router is weak: about 0.45 accuracy over 4 categories, no better than TF-IDF.)"""
    saved = joblib.load(TICKET_HEAD_FILE)
    probabilities = saved["head"].predict_proba(_bert_features([clean_text(t) for t in tickets]))
    answers = []
    for row in probabilities:
        best = int(row.argmax())
        answers.append({"category": saved["categories"][best], "confidence": round(float(row[best]), 4)})
    return answers


if __name__ == "__main__":
    samples = ["Absolutely love it, works perfectly and arrived early.",
               "Total waste of money. Stopped working after two days.",
               "It is not bad at all, quite good actually."]
    backends = ["tfidf", "bert-frozen"] + (["bert-finetuned"] if os.path.isdir(FINE_TUNED_FOLDER) else [])
    answers = {b: predict_sentiment(samples, b) for b in backends}
    for i, text in enumerate(samples):
        print("  ".join(f"{b}: {answers[b][i]['sentiment']} ({answers[b][i]['probability_positive']})" for b in backends))
        print("   ", text)
    tickets = ["The box never showed up and the tracking is stuck", "It fell apart in my hands after a week"]
    for text, answer in zip(tickets, route_ticket(tickets)):
        print(f"ticket -> {answer['category']} ({answer['confidence']})  {text}")
