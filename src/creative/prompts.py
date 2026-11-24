"""Prompt templates and fixed seeds for on-brand marketing imagery."""

import zlib

from ..config import (
    MARKETING_GUIDANCE_SCALE,
    MARKETING_IMAGE_SIZE,
    MARKETING_INFERENCE_STEPS,
    MARKETING_MODEL,
    MARKETING_STYLES,
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, extra limbs, "
    "watermark, text, logo, brand name, busy background, clutter, oversaturated, "
    "harsh shadows, wrong product, mangled details"
)

STYLE_PROMPTS = {
    "catalog": (
        "clean studio product photography, soft neutral background, diffused light, "
        "sharp focus, centered product"
    ),
    "lifestyle": (
        "warm authentic lifestyle photography, natural daylight, real home setting, "
        "relaxed and practical mood"
    ),
    "social": (
        "bright modern editorial flat lay, clean negative space, cheerful palette, "
        "social-media ready composition"
    ),
}

CATEGORY_SETTINGS = {
    "Audio": "a tidy desk with a laptop and notebook",
    "Kitchen": "a sunlit kitchen counter with a linen cloth",
    "Fitness": "a calm home workout corner with a wooden floor",
}
DEFAULT_SETTING = "a clean, neutral retail studio"

BASE_SEEDS = {
    "RM-AUDIO-01": 101,
    "RM-HOME-02": 202,
    "RM-FIT-03": 303,
}
STYLE_SEED_OFFSETS = {"catalog": 0, "lifestyle": 11, "social": 23}


def resolve_seed(product_id, style="catalog"):
    """Return a repeatable seed so an approved image can be regenerated exactly."""
    if style not in STYLE_SEED_OFFSETS:
        raise ValueError(f"Unknown marketing style: {style}")
    base = BASE_SEEDS.get(product_id)
    if base is None:
        base = zlib.crc32(str(product_id).encode("utf-8")) % 100000
    return int(base) + STYLE_SEED_OFFSETS[style]


def setting_for(product):
    """Choose a simple setting from the product category."""
    return CATEGORY_SETTINGS.get(product.get("category", ""), DEFAULT_SETTING)


def build_prompt(product, style="catalog", brand_voice=None, setting=None):
    """Turn catalog metadata into a full Stable Diffusion prompt."""
    if style not in STYLE_PROMPTS:
        raise ValueError(f"Unknown marketing style: {style}")
    if "name" not in product or "product_id" not in product:
        raise ValueError("A product needs product_id and name")

    category = product.get("category", "product").lower()
    article = "an" if category.startswith(("a", "e", "i", "o", "u")) else "a"
    features = "; ".join(product.get("features", []))
    parts = [
        f"Professional product photograph of {product['name']}, {article} {category} product.",
        product.get("details", "").strip(),
        f"Key features: {features}." if features else "",
        STYLE_PROMPTS[style] + ".",
        f"Setting: {setting or setting_for(product)}.",
        f"Brand tone: {brand_voice.strip().rstrip('.')}." if brand_voice else "",
        "Premium retail campaign, realistic materials, sharp focus, no text, no logo.",
    ]
    return " ".join(part for part in parts if part)


def build_negative_prompt(extra=None):
    """List the artifacts we do not want in an approved marketing asset."""
    return DEFAULT_NEGATIVE_PROMPT if not extra else f"{DEFAULT_NEGATIVE_PROMPT}, {extra}"


def build_spec(
    product,
    style="catalog",
    brand_voice=None,
    seed=None,
    guidance_scale=MARKETING_GUIDANCE_SCALE,
    steps=MARKETING_INFERENCE_STEPS,
    size=MARKETING_IMAGE_SIZE,
):
    """Build one complete, reproducible image request."""
    setting = setting_for(product)
    return {
        "product_id": product["product_id"],
        "product_name": product["name"],
        "category": product.get("category", "general"),
        "style": style,
        "setting": setting,
        "prompt": build_prompt(product, style, brand_voice, setting),
        "negative_prompt": build_negative_prompt(),
        "seed": resolve_seed(product["product_id"], style) if seed is None else int(seed),
        "guidance_scale": float(guidance_scale),
        "steps": int(steps),
        "width": int(size),
        "height": int(size),
        "model": MARKETING_MODEL,
    }


def build_prompt_library(catalog, styles=None, seeds=None):
    """Build the small prompt library from the same catalog Day 14 uses."""
    if "products" not in catalog:
        raise ValueError("Catalog is missing products")
    styles = tuple(MARKETING_STYLES if styles is None else styles)
    unknown = set(styles) - set(STYLE_PROMPTS)
    if unknown:
        raise ValueError(f"Unknown marketing styles: {sorted(unknown)}")

    specs = []
    for product in catalog["products"]:
        for style in styles:
            product_seeds = (seeds or {}).get(product["product_id"], {})
            specs.append(
                build_spec(
                    product,
                    style=style,
                    brand_voice=catalog.get("brand_voice"),
                    seed=product_seeds.get(style),
                )
            )
    return specs
