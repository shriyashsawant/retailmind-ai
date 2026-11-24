import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.assistant.data import load_catalog
from src.creative import (
    MarketingImageGenerator,
    build_prompt,
    build_prompt_library,
    generate_asset,
    main,
    prompt_library,
    resolve_seed,
)
from src.creative.prompts import build_negative_prompt


class FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake png")


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[FakeImage()])


class RecordingCreative:
    def __init__(self):
        self.specs = None
        self.manifest = None

    def generate_library(self, specs, manifest_path=None):
        self.specs = specs
        self.manifest = manifest_path
        return {"assets": [{"output": "fake.png"}], "device": "cpu"}


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()

    def test_library_has_nine_reproducible_specs(self):
        specs = build_prompt_library(self.catalog)
        self.assertEqual(len(specs), 9)
        self.assertEqual(
            [spec["seed"] for spec in specs],
            [101, 112, 124, 202, 213, 225, 303, 314, 326],
        )
        first = specs[0]
        self.assertIn("RetailMind Studio Wireless Headphones", first["prompt"])
        self.assertIn("clean studio product photography", first["prompt"])
        self.assertIn("Warm, concise and practical", first["prompt"])
        self.assertIn("no text, no logo", first["prompt"])
        self.assertIn("blurry", first["negative_prompt"])
        self.assertEqual(first["guidance_scale"], 7.5)
        self.assertEqual((first["width"], first["height"]), (512, 512))

    def test_prompt_uses_category_setting_and_rejects_bad_style(self):
        product = self.catalog["products"][1]
        prompt = build_prompt(product, style="lifestyle", brand_voice="Warm.")
        self.assertIn("sunlit kitchen counter", prompt)
        self.assertIn("warm authentic lifestyle photography", prompt)
        self.assertIn("a kitchen product", prompt)
        with self.assertRaises(ValueError):
            build_prompt(product, style="unknown")
        self.assertEqual(resolve_seed("NEW-PRODUCT", "social"), resolve_seed("NEW-PRODUCT", "social"))

    def test_negative_prompt_can_be_extended(self):
        self.assertIn("watermark", build_negative_prompt())
        self.assertIn("cartoon", build_negative_prompt("cartoon"))


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.spec = build_prompt_library(self.catalog, styles=("catalog",))[0]

    def test_generate_saves_image_and_passes_controls(self):
        pipeline = FakePipeline()
        seen = []

        def fake_seed(seed, device):
            seen.append((seed, device))
            return object()

        with tempfile.TemporaryDirectory() as tmp:
            creative = MarketingImageGenerator(pipeline=pipeline, output_folder=tmp, device="cpu")
            record = creative.generate(self.spec, generator_factory=fake_seed)
            self.assertEqual(Path(record["output"]).read_bytes(), b"fake png")
            self.assertEqual(seen, [(101, "cpu")])

        call = pipeline.calls[0]
        self.assertEqual(call["prompt"], self.spec["prompt"])
        self.assertEqual(call["negative_prompt"], self.spec["negative_prompt"])
        self.assertEqual(call["guidance_scale"], 7.5)
        self.assertEqual(call["num_inference_steps"], 30)
        self.assertEqual((call["width"], call["height"]), (512, 512))

    def test_generate_library_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            specs = build_prompt_library(self.catalog, styles=("catalog",))
            creative = MarketingImageGenerator(
                pipeline=FakePipeline(), output_folder=tmp, device="cpu"
            )
            artifact = creative.generate_library(
                specs,
                manifest_path=manifest,
            )
            with open(manifest, encoding="utf-8") as handle:
                saved = json.load(handle)

        self.assertEqual(len(artifact["assets"]), 3)
        self.assertEqual([item["seed"] for item in artifact["assets"]], [101, 202, 303])
        self.assertEqual(saved["model"], "runwayml/stable-diffusion-v1-5")
        self.assertEqual(saved["device"], "cpu")
        self.assertEqual(len(saved["assets"]), 3)


class ServiceTests(unittest.TestCase):
    def test_prompt_library_can_be_saved_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "library.json"
            with patch("src.creative.generator.load_pipeline", side_effect=AssertionError):
                artifact = prompt_library(output_path=output)
            with open(output, encoding="utf-8") as handle:
                saved = json.load(handle)
        self.assertEqual(len(artifact["prompts"]), 9)
        self.assertEqual(saved["prompts"][0]["product_id"], "RM-AUDIO-01")

    def test_cli_dry_run_prints_without_loading_model(self):
        buffer = io.StringIO()
        with patch(
            "src.creative.service.MarketingImageGenerator", side_effect=AssertionError
        ), redirect_stdout(buffer):
            code = main(["generate", "--dry-run", "--product", "RM-AUDIO-01", "--style", "catalog"])
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("seed 101", output)
        self.assertIn("Dry run only", output)

    def test_generate_asset_rejects_unknown_product(self):
        with self.assertRaises(ValueError):
            generate_asset(product_id="NOPE")

    def test_generate_asset_forwards_seed_override(self):
        recording = RecordingCreative()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            generate_asset(
                product_id="RM-AUDIO-01",
                all_styles=True,
                seed=900,
                generator=recording,
                manifest_path=manifest,
            )
        self.assertEqual([spec["seed"] for spec in recording.specs], [900, 901, 902])
        self.assertEqual(recording.manifest, manifest)


if __name__ == "__main__":
    unittest.main()
