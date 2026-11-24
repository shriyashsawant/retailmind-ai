from ..config import (
    CATALOG_PATH,
    REVIEW_DATA_PATH,
    SUMMARY_MAX_LENGTH,
    SUMMARY_MAX_WORDS,
    SUMMARY_MIN_LENGTH,
    SUMMARY_MODEL,
)
from .data import load_catalog, load_reviews

THEME_LABELS = {
    1: ("positive", "What customers like"),
    0: ("negative", "Common complaints"),
}


def chunk_texts(texts, max_words=SUMMARY_MAX_WORDS):
    """Pack reviews into chunks the summarization model can handle."""
    chunks = []
    current = []
    for text in texts:
        words = str(text).split()
        if not words:
            continue
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        while len(words) > max_words:
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


class ReviewSummarizer:
    """Condense many reviews into per-product theme summaries."""

    def __init__(
        self,
        pipeline=None,
        model_name=SUMMARY_MODEL,
        max_words=SUMMARY_MAX_WORDS,
        max_length=SUMMARY_MAX_LENGTH,
        min_length=SUMMARY_MIN_LENGTH,
    ):
        self._pipeline = pipeline
        self.model_name = model_name
        self.max_words = max_words
        self.max_length = max_length
        self.min_length = min_length

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "Install the summarization libraries with: pip install -r requirements.txt"
                ) from exc
            self._pipeline = hf_pipeline("summarization", model=self.model_name)
        return self._pipeline

    def summarize_texts(self, texts):
        """Run the HF summarization pipeline over one or more text chunks."""
        chunks = chunk_texts(texts, max_words=self.max_words)
        if not chunks:
            return ""

        max_length = min(
            self.max_length,
            max(self.min_length + 1, min(len(chunk.split()) for chunk in chunks) - 1),
        )
        min_length = min(self.min_length, max_length - 1)
        outputs = self.pipeline(
            chunks,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return " ".join(
            output["summary_text"].strip()
            for output in outputs
            if output.get("summary_text")
        ).strip()

    def summarize_product_reviews(self, reviews, product_name=None):
        """Summarize a product's positive and negative reviews into themes."""
        if hasattr(reviews, "to_dict"):
            reviews = reviews.to_dict("records")
        else:
            reviews = list(reviews)
        if not reviews:
            raise ValueError("No reviews supplied for this product")

        grouped = {1: [], 0: []}
        for review in reviews:
            label = int(review["label"])
            if label in grouped:
                grouped[label].append(str(review["text"]))

        themes = []
        for label, (sentiment, heading) in THEME_LABELS.items():
            if not grouped[label]:
                continue
            themes.append(
                {
                    "sentiment": sentiment,
                    "theme": heading,
                    "review_count": len(grouped[label]),
                    "summary": self.summarize_texts(grouped[label]),
                }
            )

        product_id = str(reviews[0]["product_id"])
        return {
            "product_id": product_id,
            "product_name": product_name or product_id,
            "review_count": len(reviews),
            "themes": themes,
            "summary": " ".join(
                f"{theme['theme']}: {theme['summary']}" for theme in themes
            ),
        }


def summarize_reviews(
    product_id=None,
    reviews_path=REVIEW_DATA_PATH,
    catalog_path=CATALOG_PATH,
    summarizer=None,
):
    """Return theme summaries for one product or for all reviewed products."""
    reviews = load_reviews(reviews_path)
    catalog = load_catalog(catalog_path)
    product_names = {
        product["product_id"]: product["name"] for product in catalog["products"]
    }

    if product_id is not None:
        if product_id not in set(reviews["product_id"]):
            raise ValueError(
                f"Unknown product_id {product_id!r}. "
                f"Available: {sorted(set(reviews['product_id']))}"
            )
        reviews = reviews[reviews["product_id"] == product_id]

    summarizer = summarizer or ReviewSummarizer()
    results = []
    for current_id in sorted(set(reviews["product_id"])):
        product_reviews = reviews[reviews["product_id"] == current_id]
        results.append(
            summarizer.summarize_product_reviews(
                product_reviews, product_name=product_names.get(current_id)
            )
        )
    return results
