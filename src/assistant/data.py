import json

import pandas as pd

from ..config import CATALOG_PATH, REVIEW_DATA_PATH


def load_reviews(path=REVIEW_DATA_PATH):
    """Load reviews in the amazon_polarity schema plus a local product_id column."""
    frame = pd.read_csv(path)
    missing = {"product_id", "text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Review data is missing columns: {sorted(missing)}")

    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].ne("")]
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def load_catalog(path=CATALOG_PATH):
    """Load product metadata, store policies and the brand voice."""
    with open(path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)

    missing = {"products", "store_policies"} - set(catalog)
    if missing:
        raise ValueError(f"Catalog is missing keys: {sorted(missing)}")
    return catalog


def catalog_documents(catalog):
    """Flatten catalog entries into searchable text snippets with stable ids."""
    documents = []
    for product in catalog["products"]:
        features = "; ".join(product.get("features", []))
        text = " ".join(
            [
                f"{product['name']} ({product['product_id']}).",
                f"Category: {product.get('category', 'general')}.",
                f"Price: {product.get('price', 'not listed')}.",
                product.get("details", ""),
                f"Features: {features}." if features else "",
            ]
        )
        documents.append(
            {
                "id": product["product_id"],
                "product_id": product["product_id"],
                "text": " ".join(text.split()),
            }
        )

    for policy in catalog["store_policies"]:
        documents.append(
            {
                "id": policy["id"],
                "product_id": None,
                "text": f"{policy['title']}: {policy['text']}",
            }
        )
    return documents
