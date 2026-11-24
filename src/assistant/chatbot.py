import re
from pathlib import Path

from ..config import CATALOG_PATH
from .client import LLMClient
from .data import catalog_documents, load_catalog
from .prompts import build_chat_messages, build_system_prompt

STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from", "how", "i", "in",
    "is", "it", "my", "of", "on", "or", "the", "to", "what", "when", "where",
    "with", "you", "your", "about", "there", "this", "that", "have", "has",
}


def _tokens(text):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in STOPWORDS
    }


def retrieve_context(question, documents, product_id=None, max_docs=3):
    """Pick the most relevant catalog snippets using simple token overlap."""
    query_tokens = _tokens(question)
    scored = []
    for document in documents:
        doc_product = document.get("product_id")
        if product_id and doc_product and doc_product != product_id:
            continue
        score = len(query_tokens & _tokens(document["text"]))
        if product_id and doc_product == product_id:
            score += 1
        scored.append((score, document))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    selected = [document for score, document in ranked[:max_docs] if score > 0]
    if not selected and product_id:
        selected = [
            document
            for document in documents
            if document.get("product_id") == product_id
        ][:max_docs]

    if not selected:
        return "No matching product or policy information is available."

    return "\n\n".join(
        f"[{document['id']}] {document['text']}" for document in selected
    )


class SupportChatbot:
    """Conversational support agent grounded in product metadata and policies."""

    def __init__(self, llm=None, catalog=None, catalog_path=CATALOG_PATH, max_history=10):
        self.llm = llm or LLMClient()
        self.catalog_path = Path(catalog_path)
        self.catalog = catalog or load_catalog(catalog_path)
        self.documents = catalog_documents(self.catalog)
        self.max_history = max_history
        self.history = []

    def ask(self, question, product_id=None):
        """Answer a question, remembering earlier turns in this conversation."""
        question = str(question or "").strip()
        if not question:
            raise ValueError("question must not be empty")

        if product_id and product_id not in {
            product["product_id"] for product in self.catalog["products"]
        }:
            raise ValueError(f"Unknown product_id {product_id!r}")

        context = retrieve_context(question, self.documents, product_id=product_id)
        system_prompt = build_system_prompt(
            context, self.catalog.get("brand_voice")
        )
        messages = build_chat_messages(
            system_prompt, self.history, question, max_history=self.max_history
        )
        answer = self.llm.chat(messages)

        self.history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return answer

    def reset(self):
        """Clear the conversation history."""
        self.history.clear()

    @property
    def conversation_history(self):
        return [dict(turn) for turn in self.history]
