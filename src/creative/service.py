"""Public API and CLI for the Day 15 marketing-image generator."""

import argparse
import json
from pathlib import Path

from ..assistant.data import load_catalog
from ..config import (
    CATALOG_PATH,
    MARKETING_MANIFEST_PATH,
    MARKETING_MODEL,
    MARKETING_OUTPUT_FOLDER,
    MARKETING_STYLES,
    PROMPT_LIBRARY_PATH,
)
from .generator import MarketingImageGenerator
from .prompts import build_prompt_library


def _selected_products(catalog, product_id):
    if product_id is None:
        return catalog["products"]
    products = [p for p in catalog["products"] if p["product_id"] == product_id]
    if not products:
        raise ValueError(f"Unknown product_id: {product_id}")
    return products


def prompt_library(
    catalog_path=CATALOG_PATH,
    output_path=None,
    product_id=None,
    style=None,
):
    """Build the prompt library from product metadata without loading the model."""
    catalog = load_catalog(catalog_path)
    _selected_products(catalog, product_id)
    specs = build_prompt_library(catalog, styles=(style,) if style else None)
    if product_id is not None:
        specs = [spec for spec in specs if spec["product_id"] == product_id]

    artifact = {
        "source": "product_catalog.json",
        "model": MARKETING_MODEL,
        "prompts": specs,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
    return artifact


def generate_asset(
    product_id=None,
    style=None,
    all_styles=False,
    seed=None,
    catalog_path=CATALOG_PATH,
    output_folder=MARKETING_OUTPUT_FOLDER,
    manifest_path=MARKETING_MANIFEST_PATH,
    model=MARKETING_MODEL,
    device=None,
    generator=None,
):
    """Generate one product/style asset, or the whole library."""
    catalog = load_catalog(catalog_path)
    _selected_products(catalog, product_id)
    styles = tuple(MARKETING_STYLES) if all_styles or style is None else (style,)
    specs = build_prompt_library(catalog, styles=styles)
    if product_id is not None:
        specs = [spec for spec in specs if spec["product_id"] == product_id]
    if seed is not None:
        for index, spec in enumerate(specs):
            spec["seed"] = int(seed) + index

    creative = generator or MarketingImageGenerator(
        model=model, device=device, output_folder=output_folder
    )
    return creative.generate_library(specs, manifest_path=manifest_path)


def _print_specs(specs):
    for spec in specs:
        print(f"\n{spec['product_id']} | {spec['style']} | seed {spec['seed']}")
        print(f"  setting: {spec['setting']}")
        print(f"  prompt: {spec['prompt']}")
        print(f"  negative: {spec['negative_prompt']}")
        print(
            f"  guidance {spec['guidance_scale']} | steps {spec['steps']} | "
            f"{spec['width']}x{spec['height']}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="RetailMind diffusion marketing imagery")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompts_parser = subparsers.add_parser("prompts", help="Show or save the prompt library")
    prompts_parser.add_argument("--product", help="Limit to one product_id")
    prompts_parser.add_argument("--style", help="Limit to one style")
    prompts_parser.add_argument("--output", default=None, help="Save the library as JSON")

    generate_parser = subparsers.add_parser("generate", help="Render images with Stable Diffusion")
    generate_parser.add_argument("--product", help="Limit to one product_id")
    generate_parser.add_argument("--style", default="catalog", help="Style to render")
    generate_parser.add_argument("--all-styles", action="store_true")
    generate_parser.add_argument("--seed", type=int, help="Override the fixed seed")
    generate_parser.add_argument("--output", default=MARKETING_OUTPUT_FOLDER)
    generate_parser.add_argument("--manifest", default=MARKETING_MANIFEST_PATH)
    generate_parser.add_argument("--model", default=MARKETING_MODEL)
    generate_parser.add_argument("--device", help="cuda or cpu; auto-detected by default")
    generate_parser.add_argument("--dry-run", action="store_true", help="Show specs only")

    args = parser.parse_args(argv)

    try:
        if args.command == "prompts":
            artifact = prompt_library(
                output_path=args.output, product_id=args.product, style=args.style
            )
            _print_specs(artifact["prompts"])
            if args.output:
                print(f"\nSaved prompt library -> {args.output}")
            return 0

        if args.dry_run:
            specs = build_prompt_library(
                load_catalog(), styles=tuple(MARKETING_STYLES) if args.all_styles else (args.style,)
            )
            if args.product:
                specs = [s for s in specs if s["product_id"] == args.product]
            if args.seed is not None:
                for index, spec in enumerate(specs):
                    spec["seed"] = args.seed + index
            _print_specs(specs)
            print("\nDry run only: no model downloaded and no image saved.")
            return 0

        artifact = generate_asset(
            product_id=args.product,
            style=args.style,
            all_styles=args.all_styles,
            seed=args.seed,
            output_folder=args.output,
            manifest_path=args.manifest,
            model=args.model,
            device=args.device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        print(f"Could not generate marketing imagery: {exc}")
        return 1

    print(f"Generated {len(artifact['assets'])} asset(s) on {artifact['device']}:")
    for record in artifact["assets"]:
        print(f"  {record['product_id']} | {record['style']} | seed {record['seed']} -> {record['output']}")
    print(f"Manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
