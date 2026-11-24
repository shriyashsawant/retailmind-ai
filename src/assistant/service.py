import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import CATALOG_PATH, REVIEW_DATA_PATH, SUMMARIES_PATH
from .chatbot import SupportChatbot
from .client import LLMConfigurationError, LLMError
from .summarizer import ReviewSummarizer, summarize_reviews


def build_summaries(
    product_id=None,
    output_path=SUMMARIES_PATH,
    reviews_path=REVIEW_DATA_PATH,
    catalog_path=CATALOG_PATH,
    summarizer=None,
):
    """Generate review theme summaries and persist them as JSON."""
    summaries = summarize_reviews(
        product_id=product_id,
        reviews_path=reviews_path,
        catalog_path=catalog_path,
        summarizer=summarizer,
    )
    artifact = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "amazon_polarity-style review sample",
        "summaries": summaries,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
    return artifact


_CHATBOT = None


def get_chatbot(catalog_path=CATALOG_PATH):
    """Return the shared support bot so a CLI session keeps its history."""
    global _CHATBOT
    if _CHATBOT is None or _CHATBOT.catalog_path != Path(catalog_path):
        _CHATBOT = SupportChatbot(catalog_path=catalog_path)
    return _CHATBOT


def answer(question, product_id=None, chatbot=None):
    """Ask the support assistant one question."""
    return (chatbot or get_chatbot()).ask(question, product_id=product_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RetailMind LLM assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summaries", help="Summarize reviews by product")
    summary_parser.add_argument("--product", help="Limit to one product_id")
    summary_parser.add_argument("--output", default=str(SUMMARIES_PATH))
    summary_parser.add_argument("--model", help="Hugging Face summarization model")

    ask_parser = subparsers.add_parser("ask", help="Ask one support question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--product", help="Ground the answer in one product")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive support chat")
    chat_parser.add_argument("--product", help="Ground the chat in one product")

    args = parser.parse_args(argv)

    if args.command == "summaries":
        summarizer = ReviewSummarizer(model_name=args.model) if args.model else None
        try:
            artifact = build_summaries(
                product_id=args.product,
                output_path=args.output,
                summarizer=summarizer,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"Could not build summaries: {exc}")
            return 1

        for summary in artifact["summaries"]:
            print(f"\n{summary['product_name']} ({summary['product_id']})")
            for theme in summary["themes"]:
                print(f"  {theme['theme']} ({theme['review_count']} reviews)")
                print(f"    {theme['summary']}")
        print(f"\nSaved summaries -> {args.output}")
        return 0

    try:
        chatbot = get_chatbot()
    except (LLMConfigurationError, OSError) as exc:
        print(f"Could not start the support bot: {exc}")
        return 1

    if args.command == "ask":
        try:
            print(chatbot.ask(args.question, product_id=args.product))
        except (LLMError, LLMConfigurationError, ValueError) as exc:
            print(f"Could not answer: {exc}")
            return 1
        return 0

    print("RetailMind support chat. Type 'quit' or 'exit' to leave.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        try:
            print(f"\nAssistant: {chatbot.ask(question, product_id=args.product)}\n")
        except (LLMError, LLMConfigurationError, ValueError) as exc:
            print(f"Error: {exc}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
