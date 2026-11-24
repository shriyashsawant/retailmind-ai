"""Stable Diffusion image generation with lazy model loading and fixed seeds."""

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import MARKETING_MANIFEST_PATH, MARKETING_MODEL, MARKETING_OUTPUT_FOLDER

_PIPELINE = None
_PIPELINE_KEY = None


def select_device(requested=None):
    """Use CUDA when present, otherwise allow a slow CPU run."""
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(model=MARKETING_MODEL, device=None):
    """Load and cache the pretrained pipeline only when an image is requested."""
    global _PIPELINE, _PIPELINE_KEY
    device = select_device(device)
    key = (model, device)
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Day 15 needs diffusers and torch. Install with: pip install diffusers"
        ) from exc

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
    _PIPELINE = pipeline.to(device)
    _PIPELINE_KEY = key
    return _PIPELINE


def make_generator(seed, device=None):
    """Create the seeded random generator diffusers uses for repeatable images."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Seeded generation needs torch") from exc
    generator = torch.Generator(device=select_device(device))
    generator.manual_seed(int(seed))
    return generator


class MarketingImageGenerator:
    """Generate approved-style product assets from the prompt library."""

    def __init__(
        self,
        pipeline=None,
        model=MARKETING_MODEL,
        device=None,
        output_folder=MARKETING_OUTPUT_FOLDER,
        pipeline_factory=None,
    ):
        self.model = model
        self.device = select_device(device)
        self.output_folder = Path(output_folder)
        self.pipeline = pipeline
        self.pipeline_factory = pipeline_factory

    @property
    def active_pipeline(self):
        if self.pipeline is None:
            if self.pipeline_factory is not None:
                self.pipeline = self.pipeline_factory()
            else:
                self.pipeline = load_pipeline(self.model, self.device)
        return self.pipeline

    def generate(self, spec, output_path=None, generator_factory=make_generator):
        """Render one spec and return a record with its prompt, seed and path."""
        if output_path is None:
            filename = f"{spec['product_id']}_{spec['style']}_seed{spec['seed']}.png"
            output_path = self.output_folder / filename
        path = Path(output_path)
        generator = generator_factory(int(spec["seed"]), self.device)

        result = self.active_pipeline(
            prompt=spec["prompt"],
            negative_prompt=spec["negative_prompt"],
            guidance_scale=spec["guidance_scale"],
            num_inference_steps=spec["steps"],
            width=spec["width"],
            height=spec["height"],
            generator=generator,
        )
        image = result.images[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

        keys = (
            "product_id",
            "product_name",
            "style",
            "setting",
            "prompt",
            "negative_prompt",
            "seed",
            "guidance_scale",
            "steps",
            "width",
            "height",
        )
        record = {key: spec[key] for key in keys}
        record.update({"model": self.model, "device": self.device, "output": str(path)})
        return record

    def generate_library(self, specs, manifest_path=MARKETING_MANIFEST_PATH):
        """Render every spec and persist a manifest of exactly what was generated."""
        records = [self.generate(spec) for spec in specs]
        artifact = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self.model,
            "device": self.device,
            "assets": records,
        }
        if manifest_path is not None:
            path = Path(manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(artifact, handle, indent=2)
        return artifact
