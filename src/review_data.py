"""Days 12 and 13: get product reviews, clean them, and (Day 13) relabel some as support tickets.

The reviews come from the public Hugging Face dataset "amazon_polarity" (label 1 = positive, 0 = negative).
The first time you run this it downloads ONE file of that dataset (the 400,000-review test file, about
117 MB) into your Hugging Face cache; after that it is read from disk. We then take a small, balanced sample.

Needs:   pip install huggingface_hub pyarrow
"""
import re

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE

DATASET = "fancyzhx/amazon_polarity"
DATA_FILE = "amazon_polarity/test-00000-of-00001.parquet"      # about 117 MB, 400,000 reviews
SENTIMENTS = ["negative", "positive"]                          # label 0 and label 1


def clean_text(text):
    """Remove the clutter that is not language: HTML tags, web addresses, and extra spaces."""
    text = re.sub(r"<[^>]+>", " ", str(text))                  # <br /> and other HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)         # web addresses
    return re.sub(r"\s+", " ", text).strip()                   # runs of spaces and new lines become one space


def load_reviews(train_per_class=2000, test_per_class=500):
    """A balanced sample of cleaned reviews, split into train and test. Returns four lists/arrays:
    train_texts, test_texts, train_labels, test_labels."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset")
    table = pd.read_parquet(path, columns=["label", "title", "content"])

    random = np.random.RandomState(RANDOM_STATE)
    parts = []
    for label in (0, 1):
        rows = np.where(table["label"].to_numpy() == label)[0]
        chosen = random.choice(rows, train_per_class + test_per_class, replace=False)
        parts.append(table.iloc[chosen].assign(label=label))
    sample = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)       # shuffle so the classes are mixed

    texts = (sample["title"].fillna("") + ". " + sample["content"].fillna("")).map(clean_text).tolist()
    labels = sample["label"].to_numpy()
    cut = 2 * train_per_class
    return texts[:cut], texts[cut:], labels[:cut], labels[cut:]


# --- Day 13: support tickets, relabelled from reviews ---------------------------------------------
# A few categories a support team might route to. A review is put in a category when it contains one of its
# trigger words; the first category that matches wins; reviews that match none are skipped.
TICKET_WORDS = {
    "shipping": ["shipping", "shipped", "delivery", "delivered", "arrived", "package", "late"],
    "quality": ["broke", "broken", "defective", "cheap", "stopped working", "flimsy", "fell apart"],
    "refund": ["refund", "return", "returned", "money back", "waste of money", "overpriced"],
    "how-to": ["instructions", "manual", "install", "setup", "set up", "figure out", "confusing"],
}


def make_ticket_data(texts):
    """Turn reviews into (text, category) pairs. The trigger words are removed from the text, so a model has to
    learn from the context and not just spot the word that was used to make the label."""
    patterns = {cat: re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
                for cat, words in TICKET_WORDS.items()}
    all_triggers = re.compile("|".join(p.pattern for p in patterns.values()), re.I)     # every trigger word of every category
    kept_texts, categories = [], []
    for text in texts:
        for category, pattern in patterns.items():
            if pattern.search(text):
                kept_texts.append(re.sub(r"\s+", " ", all_triggers.sub(" ", text)).strip())
                categories.append(category)
                break
    return kept_texts, np.array(categories)
