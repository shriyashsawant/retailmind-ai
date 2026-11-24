"""RetailMind AI — Day 14 LLM assistant (review summaries + support chat)."""

from .chatbot import SupportChatbot, retrieve_context
from .client import LLMClient, LLMConfigurationError, LLMError, detect_provider
from .prompts import build_chat_messages, build_system_prompt
from .service import answer, build_summaries, get_chatbot
from .summarizer import ReviewSummarizer, chunk_texts, summarize_reviews

__all__ = [
    "LLMClient",
    "LLMConfigurationError",
    "LLMError",
    "ReviewSummarizer",
    "SupportChatbot",
    "answer",
    "build_chat_messages",
    "build_summaries",
    "build_system_prompt",
    "chunk_texts",
    "detect_provider",
    "get_chatbot",
    "retrieve_context",
    "summarize_reviews",
]
