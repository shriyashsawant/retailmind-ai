# Day 15 increment: diffusion marketing imagery

## What was added
- `src/creative/prompts.py`: builds a prompt from each product's **category, features and details**, adds one of three styles (`catalog`, `lifestyle`, `social`), picks a setting from the category, and includes the catalog's brand voice. It also holds the negative prompt, guidance scale, steps, image size and fixed seeds.
- `src/creative/generator.py`: lazily loads `StableDiffusionPipeline` from `runwayml/stable-diffusion-v1-5` with `diffusers`, selects CUDA when available, seeds torch's generator, calls the pipeline with negative prompt and guidance scale, saves the image and writes a manifest.
- `src/creative/service.py` and `src/creative/__main__.py`: the creative module's public API and CLI: `python -m src.creative prompts` and `python -m src.creative generate` (with `--dry-run`, `--product`, `--style`, `--all-styles`, `--seed`, `--device`).
- `data/prompt_library.json`: nine checked-in prompt specs (3 products × 3 styles) with seeds `101/112/124`, `202/213/225`, `303/314/326`. This is the small, reproducible prompt library an approver can sign off.
- `tests/test_creative.py`: nine offline tests with a fake diffusion pipeline and image, covering prompts, seeds, negative prompts, guidance scale, image saving, manifest writing and the CLI. No model download or GPU is needed for the tests.
- `notebooks/12_day15_diffusion_marketing.ipynb`: saved walkthrough of the prompt library, controls, seed logic and generator calls.
- `requirements.txt`: adds `diffusers>=0.31,<0.36` and pins `huggingface_hub>=0.34,<1` so it stays compatible with the Transformers version used on Day 14.

## How it connects to earlier days
- The product metadata and brand voice come from the same `data/raw/product_catalog.json` Day 14 uses for the support chatbot. One catalog now drives both conversation and creative output.
- The `*_service.py` pattern continues: `generate_asset()` hides the heavy diffusers work behind a small function, just as `predict()` and `answer()` do.
- The fixed-seed idea follows the reproducible `random_state=42` habit from the churn days: an approved image can be regenerated from its seed instead of being lost.
- Days 1–14 were left untouched apart from adding the Day 15 constants in `src/config.py`, the two dependency pins and the links in the README.

## The three teaching sessions
1. **Diffusion in one minute.** The model starts with random noise and repeatedly denoises it toward an image that matches the text prompt. `num_inference_steps` is how many denoising passes we allow; 30 is a reasonable quality/speed compromise.
2. **Controlling the result.** The positive prompt says what should be in the picture; the **negative prompt** says what must not appear (blur, text, logos, distorted details). **Guidance scale** controls how strongly the model obeys the prompt: 7.5 keeps it on-brief without making the image look artificially pushed.
3. **Reproducibility and a prompt library.** A fixed `seed` fixes the starting noise, so the same prompt and seed reproduce the same asset. The nine saved specs in `data/prompt_library.json` are the approved creative briefs; changing a seed is an explicit decision, not an accident.

## What happened in this increment
1. The catalog was read and three style templates were applied to every product.
2. Category-specific settings (desk, kitchen, workout corner) and the brand voice were added to each prompt.
3. Negative prompts and a 7.5 guidance scale were added to protect quality and brand consistency.
4. Fixed seeds were assigned per product and style, then saved as a nine-entry prompt library.
5. The generator was built with lazy model loading, offline fake-pipeline tests and a manifest so every saved image can be traced to its exact prompt and seed.

## Results
No real image was rendered in this session: the workspace has no GPU, and Stable Diffusion v1-5 is about 4 GB to download. The code loads the real diffusers pipeline on the first `generate` call, and the tests use a fake pipeline. On a GPU runtime:

```bash
python -m src.creative generate --product RM-AUDIO-01 --style catalog
python -m src.creative generate --all-styles
```

would write PNGs to `reports/marketing/` and a traceable manifest to `reports/marketing_assets.json`. Both paths are gitignored; the prompt library is not.

## Business value
A product manager can turn the existing catalog into studio, lifestyle and social assets in minutes instead of booking a shoot for every SKU. The prompt library and seeds make approved looks repeatable across campaigns, and the manifest gives the team a record of exactly what was generated.

## Open blockers / honest notes
1. **A GPU is strongly recommended.** On CPU, 30 steps at 512×512 can take minutes per image. The CLI auto-detects CUDA and falls back to CPU with a slower run.
2. **Diffusers is pinned below 0.36** because newer releases require `huggingface_hub>=1`, which conflicts with the Transformers version pinned on Day 14. Raising both pins together is a future upgrade task.
3. **The model can still get small details wrong** (hands, text, exact product shape). The negative prompt reduces this but does not eliminate it; a human should approve assets before publication.
4. **Retrieval is not used here** — prompts come from structured catalog fields. This is deliberate: it is predictable and auditable.
5. **Next iteration (Try-It-Out Challenge):** use img2img to create variations of an approved shot, then assemble a short promo video clip. Neither is implemented in this increment.
