# Day 13 increment: BERT for review sentiment and ticket routing

## What was added
- `src/train_day13.py`: compares three sentiment models on the same 4,000 train / 1,000 test Amazon reviews used on Day 12: TF-IDF + logistic regression (Day 12 baseline), frozen DistilBERT features + logistic regression, and fine-tuned DistilBERT. Also builds a 4-category support-ticket router.
- `src/sentiment_service.py`: `predict_sentiment()` (default = fine-tuned model if present, else TF-IDF) and `route_ticket()`. This replaces the classical-only sentiment module in the pipeline.
- `notebooks/10_day13_bert_text.ipynb`: reads the saved results and tries the module; `models/day13_results.json`, `models/sentiment_bert_head.joblib`, `models/ticket_router.joblib`.

## How it connects to earlier days
- Day 12 (`src/train_day12.py`, `src/review_data.py`) supplies the data loader, the cleaned reviews and the TF-IDF baseline that BERT must beat.
- The bootstrap comparison and "report the honest result" habit come from Days 9-11 (multi-seed checks, controls).
- The service function follows the same pattern as the earlier `*_service.py` modules (`service.py`, `ann_service.py`, `catalog_service.py`, `demand_service.py`).

## Results (1,000 held-out reviews)
| Model | Accuracy |
| --- | ---: |
| TF-IDF + logistic regression (Day 12) | 0.870 |
| DistilBERT frozen + logistic regression | 0.867 (tie, range -0.026 to +0.020) |
| DistilBERT fine-tuned (1 pass) | 0.902 (+0.032 over TF-IDF, 95% range +0.011 to +0.054) |

Cost: about 113 vs 34,600 reviews per second and 268 MB vs 1.2 MB. Ticket routing: no BERT lift (0.455 vs 0.4925 for TF-IDF).

## Business value
Fine-tuned BERT classifies about 3 more reviews in 100 correctly, which helps when each mistake costs a missed complaint or a mis-sorted product insight. The gain is modest and the model is about 300 times slower, so for large volumes on a CPU the TF-IDF baseline is still a reasonable default. The service keeps both behind one function.

## Open blockers / honest notes (for doubt-clearing)
1. **The course lab has no `torch`/`transformers`**, so the BERT training and the BERT backends cannot run there. Training was done on a laptop; the lab shows the saved results and runs the TF-IDF path.
2. **Version compatibility:** models saved with a newer scikit-learn failed to load in the lab's scikit-learn 1.5.2 (`'LogisticRegression' object has no attribute 'multi_class'`). `_load()` in `sentiment_service.py` now handles it.
3. The fine-tuned model folder (about 270 MB) is not stored in git; `python -m src.train_day13` creates it.
4. Fine-tuning was run once (seed 42), so its exact score is uncertain; treat 0.902 as "about 0.90".
5. The ticket categories come from my own keyword rules with the keywords removed, so the router is weak and should not be used as a real router.
6. Next increment (Day 14: LLM summaries and a support chatbot) needs an LLM API key.

## Run it
```
python -c "from src.sentiment_service import predict_sentiment; print(predict_sentiment(['Love it, works perfectly'], backend='tfidf'))"
```
