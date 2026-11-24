import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.assistant import (
    LLMClient,
    LLMConfigurationError,
    ReviewSummarizer,
    SupportChatbot,
    build_chat_messages,
    build_summaries,
    build_system_prompt,
    chunk_texts,
    detect_provider,
    retrieve_context,
    summarize_reviews,
)
from src.assistant.data import catalog_documents, load_catalog, load_reviews
from src.assistant.service import main


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, chunks, **kwargs):
        self.calls.append((list(chunks), kwargs))
        return [
            {"summary_text": f"summary of {len(chunk.split())} words"}
            for chunk in chunks
        ]


class RecordingLLM:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies or ["reply one", "reply two"]

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


class DataTests(unittest.TestCase):
    def test_bundled_data_loads(self):
        reviews = load_reviews()
        catalog = load_catalog()
        self.assertEqual(len(reviews), 30)
        self.assertEqual(set(reviews["label"]), {0, 1})
        self.assertEqual(len(catalog["products"]), 3)
        self.assertEqual(len(catalog_documents(catalog)), 7)


class PromptTests(unittest.TestCase):
    def test_system_prompt_is_grounded(self):
        prompt = build_system_prompt("Price: $89", "Friendly and brief")
        self.assertIn("Friendly and brief", prompt)
        self.assertIn("Price: $89", prompt)
        self.assertIn("I don't have that information", prompt)
        self.assertIn("Never invent prices", prompt)

    def test_history_is_trimmed(self):
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"turn {index}",
            }
            for index in range(12)
        ]
        messages = build_chat_messages("SYSTEM", history, "new question", max_history=2)
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(messages[1]["content"], "turn 8")
        self.assertEqual(messages[-1], {"role": "user", "content": "new question"})
        self.assertEqual(len(messages), 6)


class LLMClientTests(unittest.TestCase):
    def test_key_comes_from_environment_and_is_hidden(self):
        def transport(messages, model, max_tokens, temperature):
            self.assertEqual(model, "gpt-4o-mini")
            self.assertEqual(messages[0]["content"], "Hello")
            return "Hi there"

        client = LLMClient(
            environ={"OPENAI_API_KEY": "sk-secret-value"}, transport=transport
        )
        self.assertEqual(client.provider, "openai")
        self.assertNotIn("sk-secret-value", repr(client))
        self.assertEqual(client.chat([{"role": "user", "content": "Hello"}]), "Hi there")

    def test_provider_detection(self):
        self.assertEqual(
            detect_provider({"ANTHROPIC_API_KEY": "test-key"}), "anthropic"
        )
        self.assertEqual(
            detect_provider(
                {"RETAILMIND_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}
            ),
            "openai",
        )
        with self.assertRaises(LLMConfigurationError):
            detect_provider({})

    def test_missing_key_raises_configuration_error(self):
        with self.assertRaises(LLMConfigurationError):
            LLMClient(environ={})

    def test_explicit_model_overrides_default(self):
        client = LLMClient(
            environ={"ANTHROPIC_API_KEY": "test-key"},
            model="custom-model",
            transport=lambda **kwargs: "ok",
        )
        self.assertEqual(client.model, "custom-model")

    def test_openai_sdk_call_path(self):
        class FakeOpenAI:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="sdk answer"))
                    ]
                )

        client = LLMClient(environ={"OPENAI_API_KEY": "test-key"})
        fake_sdk = FakeOpenAI()
        client._sdk_client = fake_sdk
        messages = [{"role": "user", "content": "Hello"}]
        self.assertEqual(client.chat(messages), "sdk answer")
        self.assertEqual(fake_sdk.kwargs["messages"], messages)
        self.assertEqual(fake_sdk.kwargs["model"], "gpt-4o-mini")

    def test_anthropic_sdk_call_path_separates_system(self):
        class FakeAnthropic:
            def __init__(self):
                self.messages = self

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="anthropic answer")]
                )

        client = LLMClient(environ={"ANTHROPIC_API_KEY": "test-key"})
        fake_sdk = FakeAnthropic()
        client._sdk_client = fake_sdk
        messages = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "Hello"},
        ]
        self.assertEqual(client.chat(messages), "anthropic answer")
        self.assertEqual(fake_sdk.kwargs["system"], "SYSTEM")
        self.assertEqual(fake_sdk.kwargs["messages"], [messages[1]])
        self.assertIn("temperature", fake_sdk.kwargs)


class SummarizerTests(unittest.TestCase):
    def test_long_reviews_are_chunked(self):
        chunks = chunk_texts([" ".join(["word"] * 250)], max_words=100)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk.split()) <= 100 for chunk in chunks))

    def test_reviews_become_positive_and_negative_themes(self):
        pipeline = FakePipeline()
        reviews = [
            {"product_id": "RM-AUDIO-01", "text": "Great sound and battery.", "label": 1},
            {"product_id": "RM-AUDIO-01", "text": "Soft ear cushions.", "label": 1},
            {"product_id": "RM-AUDIO-01", "text": "Connection drops often.", "label": 0},
        ]
        summary = ReviewSummarizer(pipeline=pipeline).summarize_product_reviews(
            reviews, product_name="Headphones"
        )

        self.assertEqual(summary["product_name"], "Headphones")
        self.assertEqual(summary["review_count"], 3)
        self.assertEqual(
            [theme["sentiment"] for theme in summary["themes"]],
            ["positive", "negative"],
        )
        self.assertEqual(
            pipeline.calls[0][0], ["Great sound and battery. Soft ear cushions."]
        )
        self.assertFalse(pipeline.calls[0][1]["do_sample"])
        self.assertIn("What customers like", summary["summary"])

    def test_facade_filters_by_product_and_uses_catalog_name(self):
        results = summarize_reviews(
            product_id="RM-HOME-02", summarizer=ReviewSummarizer(pipeline=FakePipeline())
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["product_name"], "RetailMind Brew Daily Coffee Maker"
        )
        self.assertEqual(results[0]["review_count"], 10)

    def test_unknown_product_raises(self):
        with self.assertRaises(ValueError):
            summarize_reviews(
                product_id="RM-UNKNOWN",
                summarizer=ReviewSummarizer(pipeline=FakePipeline()),
            )


class ChatbotTests(unittest.TestCase):
    def test_retrieval_prefers_matching_product_and_policies(self):
        documents = catalog_documents(load_catalog())
        context = retrieve_context(
            "How long does the battery last?", documents, product_id="RM-AUDIO-01"
        )
        self.assertIn("RM-AUDIO-01", context)
        self.assertIn("40-hour battery", context)

    def test_chatbot_keeps_history_and_grounds_context(self):
        llm = RecordingLLM()
        bot = SupportChatbot(llm=llm)
        first = bot.ask("How long does the battery last?", product_id="RM-AUDIO-01")
        second = bot.ask("And what is the return window?")

        self.assertEqual((first, second), ("reply one", "reply two"))
        self.assertEqual(len(bot.conversation_history), 4)
        self.assertIn("40-hour battery", llm.calls[0][0]["content"])
        self.assertIn("30 days", llm.calls[1][0]["content"])
        self.assertIn("reply one", llm.calls[1][2]["content"])
        self.assertEqual(llm.calls[1][-1]["content"], "And what is the return window?")

    def test_unknown_information_stays_ungrounded(self):
        llm = RecordingLLM(replies=["I don't have that information."])
        bot = SupportChatbot(llm=llm)
        bot.ask("Do you sell drones?")
        self.assertIn(
            "No matching product or policy information is available.",
            llm.calls[0][0]["content"],
        )

    def test_reset_clears_history(self):
        bot = SupportChatbot(llm=RecordingLLM())
        bot.ask("What is the price?")
        bot.reset()
        self.assertEqual(bot.conversation_history, [])


class ServiceTests(unittest.TestCase):
    def test_build_summaries_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "summaries.json"
            artifact = build_summaries(
                product_id="RM-FIT-03",
                output_path=output,
                summarizer=ReviewSummarizer(pipeline=FakePipeline()),
            )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved, artifact)
            self.assertEqual(saved["summaries"][0]["product_id"], "RM-FIT-03")

    def test_ask_cli_prints_answer(self):
        bot = SupportChatbot(llm=RecordingLLM(replies=["The mat is $38."]))
        buffer = io.StringIO()
        with patch("src.assistant.service.get_chatbot", return_value=bot):
            with redirect_stdout(buffer):
                status = main(["ask", "How much is the yoga mat?"])
        self.assertEqual(status, 0)
        self.assertIn("The mat is $38.", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
