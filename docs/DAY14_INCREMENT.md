# Day 14 increment: LLM summaries and a support chatbot

## What was added
- `src/assistant/summarizer.py`: groups `amazon_polarity`-style reviews by product and sentiment, packs them into chunks of at most 180 words, and runs the Hugging Face `pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")` (`max_length=64`, `min_length=12`, `do_sample=False`). Chunk summaries are joined into two themes per product: **What customers like** and **Common complaints**, each with a review count.
- `src/assistant/client.py`: one thin `LLMClient` for OpenAI **or** Anthropic. The provider and API key come only from the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `RETAILMIND_LLM_PROVIDER`, `RETAILMIND_LLM_MODEL`); no key is written to the repo and `repr(client)` masks it. A transport can be injected for tests.
- `src/assistant/prompts.py`: the system prompt fixes the brand voice and grounding rules — answer only from the supplied product/policy text, never invent details, use a fixed fallback line when information is missing, and offer a human specialist for account, damage and warranty issues.
- `src/assistant/chatbot.py`: picks up to three catalog snippets by simple word overlap, wraps them in the system prompt, keeps the last 10 conversation turns, and stores both the user and assistant turn after every answer. `reset()` clears the conversation.
- `src/assistant/service.py` and `src/assistant/__main__.py`: the first assistant module, exposed as `python -m src.assistant summaries`, `ask` and `chat`, plus the Python API `summarize_reviews()` and `answer()`.
- Data: `data/raw/amazon_reviews_sample.csv` (30 reviews: 10 per product, 6 positive and 4 negative each, with a locally added `product_id`) and `data/raw/product_catalog.json` (3 products, 4 store policies, brand voice).
- Tests and output: `tests/test_assistant.py` (19 offline tests using a fake pipeline and recording LLM/SDK objects) and `reports/review_summaries.json` (produced by a real run of the model).
- Learning material: `notebooks/11_day14_llm_assistant.ipynb` (saved walkthrough of the data, summaries, chunking, prompt and chatbot) and a plain-English Day 14 chapter in `docs/LEARNING_GUIDE.md`.

## How it connects to earlier days
- Days 12 and 13 (`src/review_data.py`, `src/sentiment_service.py`) classify **individual** reviews as positive or negative and route support tickets. Day 14 uses the same `amazon_polarity` label convention at a different level: it summarises **many** reviews for one product and then answers questions about that product.
- The `*_service.py` pattern from earlier days is kept: a small public function (`summarize_reviews()`, `answer()`) hides the heavy work behind a thin facade.
- The product metadata and policies are the RetailMind catalog; Day 15 will reuse the same descriptions and brand voice for on-brand marketing imagery.
- The Day 1 churn pipeline and the Day 2–13 models and services were not changed.

## What happened in this increment
1. Reviews were loaded and grouped by product and sentiment. Because the summarization model has a limited input window, each group was split into ≤180-word chunks.
2. Every chunk was condensed by the HF pipeline; the resulting sentences were joined into a theme summary with counts. All three products were written to `reports/review_summaries.json`.
3. The LLM API was put behind a single client that reads credentials from the environment and supports both common providers, so switching provider does not touch the chatbot.
4. Prompt templates were added so answers stay on brand and are grounded in the catalog. Unknown questions get the fallback answer instead of a made-up one.
5. The chatbot was given conversation history and a CLI. Tests were added so the whole flow can be checked without an API key or network.

## Results
A real run over the 30 bundled reviews produced, for example:

```text
RetailMind Studio Wireless Headphones (RM-AUDIO-01)
  What customers like (6 reviews)
    Sound is crisp and the bass is rich without muddying vocals. Battery lasts
    all week on a single charge. Paired with laptop in seconds.
  Common complaints (4 reviews)
    Bluetooth connection drops intermittently. Touch controls are too
    sensitive. The ear pads started peeling after a few months.
```

The other two products (coffee maker, yoga mat) produced equally product-specific positive and negative themes. The chatbot calls were verified with fake LLM/SDK objects; a live answer needs a real API key.

## Business value
Instead of reading dozens of reviews by hand, a product team gets the main praise and complaint themes in seconds. A support bot with the catalog and policies in its prompt can answer routine product questions at any hour and escalate the rest, cutting response time and workload.

## Open blockers / honest notes (for doubt-clearing)
1. **The chatbot needs an LLM API key** in the environment; no key is stored in the repo. Summaries do not need one.
2. **Retrieval is simple word overlap**, not embeddings or a vector database. With a much larger catalog, a real retriever would be the next improvement.
3. **The review sample is small** (30 curated rows in the `amazon_polarity` schema). To use the full public dataset, keep `text`, `label` and `product_id` in the CSV; the first HF download is about 117 MB.
4. **Version pins matter:** Transformers is pinned to `<5` because v5 removed the `summarization` pipeline, and Anthropic to `<1` because v2 dropped the `temperature` argument.
5. **Summaries are generated, not audited.** They can compress or merge claims, so a human should check them before publishing to customers.
6. Next increment (Day 15: on-brand marketing imagery with Stable Diffusion) will use the same catalog and brand voice.

## Run it
```bash
python -m src.assistant summaries
export OPENAI_API_KEY="..."   # or ANTHROPIC_API_KEY
python -m src.assistant ask "How long does the battery last?" --product RM-AUDIO-01
python -m src.assistant chat
python -m unittest discover -s tests
```
