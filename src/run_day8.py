"""Day 8: measure the segmentation and Grad-CAM steps and save pictures for the catalog team.

Run from the main project folder (needs models/catalog_cnn.keras from src.train_day6):
    python -m src.run_day8
    python -m src.run_day8 --photo path/to/photo.jpg      # also cut a product out of a colour photo using the saved detector box

What it checks, on the first 1,000 Fashion-MNIST test pictures:
  1. Does removing the background change what the classifier says? (accuracy before vs after)
  2. Is the Grad-CAM heat on the product? (share of heat inside the mask, for right and wrong answers)
"""
import argparse
import json
import os

import cv2
import numpy as np

from src.config import DETECTIONS_FILE
from src.explain import grad_cam, heat_inside_mask, overlay
from src.segmentation import mask_area_share, put_on_background, segment_plain_background
from src.train_day6 import FASHION_CLASSES, load_fashion_mnist
from src.vision import CatalogVision

HOW_MANY = 1000
RESULTS_FILE = "models/day8_results.json"
GRID_IMAGE = "reports/day8_catalog_grid.png"
CUTOUT_IMAGE = "reports/day8_photo_cutout.jpg"


def predictions_for(vision, pictures):
    ready = (pictures / 255.0).astype("float32")[..., np.newaxis]
    return vision.model.predict(ready, verbose=0).argmax(axis=1)


def save_grid(vision, pictures, labels, path, how_many=8):
    """Rows: original, mask, product on white, Grad-CAM. One column per picture. Enlarged 4x so it is readable."""
    columns = []
    for picture, real in zip(pictures[:how_many], labels[:how_many]):
        result = vision.analyze(picture)
        tiles = [cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR),
                 cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR),
                 cv2.cvtColor(result["clean"], cv2.COLOR_GRAY2BGR),
                 result["explanation"]]
        tiles = [cv2.resize(t, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST) for t in tiles]
        label = f"{result['category']} {result['confidence']:.2f}"
        colour = (0, 160, 0) if result["category"] == FASHION_CLASSES[int(real)] else (0, 0, 220)
        banner = np.full((22, tiles[0].shape[1], 3), 255, np.uint8)
        cv2.putText(banner, label, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        columns.append(np.vstack([banner] + tiles))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.hstack(columns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", help="a colour photo to cut a product out of")
    args = parser.parse_args()

    _, _, x_test, y_test = load_fashion_mnist()
    pictures = (x_test[:HOW_MANY, :, :, 0] * 255).astype("uint8")
    labels = y_test[:HOW_MANY].astype(int)

    vision = CatalogVision()
    original_guess = predictions_for(vision, pictures)

    masks = [segment_plain_background(p) for p in pictures]
    shares = np.array([mask_area_share(m) for m in masks])
    # background removed, then put back on BLACK, the colour the classifier was trained on
    cleaned = np.stack([put_on_background(p, m, 0) for p, m in zip(pictures, masks)])
    cleaned_guess = predictions_for(vision, cleaned)
    # background removed and put on WHITE, as the catalog team would publish it
    on_white = np.stack([put_on_background(p, m, 255) for p, m in zip(pictures, masks)])
    white_guess = predictions_for(vision, on_white)

    heat_shares, right = [], []
    for picture, mask, guess, real in zip(pictures, masks, original_guess, labels):
        heatmap, _ = grad_cam(vision.model, picture / 255.0)
        heat_shares.append(heat_inside_mask(heatmap, mask))
        right.append(bool(guess == real))
    heat_shares, right = np.array(heat_shares), np.array(right)

    results = {
        "pictures_checked": HOW_MANY,
        "accuracy_original": round(float((original_guess == labels).mean()), 4),
        "accuracy_background_removed_on_black": round(float((cleaned_guess == labels).mean()), 4),
        "accuracy_background_removed_on_white": round(float((white_guess == labels).mean()), 4),
        "same_answer_after_cleaning_on_black": round(float((original_guess == cleaned_guess).mean()), 4),
        "mask_share_of_picture_mean": round(float(shares.mean()), 3),
        "empty_masks": int((shares == 0).sum()),
        "heat_on_product_mean": round(float(heat_shares.mean()), 3),
        "heat_on_product_when_right": round(float(heat_shares[right].mean()), 3),
        "heat_on_product_when_wrong": round(float(heat_shares[~right].mean()), 3) if (~right).any() else None,
        "wrong_answers": int((~right).sum()),
    }

    save_grid(vision, pictures, labels, GRID_IMAGE)
    print(json.dumps(results, indent=2))

    if args.photo:
        with open(DETECTIONS_FILE) as f:
            objects = json.load(f)["objects"]
        biggest = max(objects, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
        cutout, _ = vision.cut_out_from_photo(args.photo, box=biggest["box"])
        cv2.imwrite(CUTOUT_IMAGE, cutout)
        results["photo_demo"] = {"box_label": biggest["label"], "output": CUTOUT_IMAGE,
                                 "note": "GrabCut inside the saved YOLO box on the Ultralytics sample photo; not a catalog image"}
        print("saved", CUTOUT_IMAGE)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", GRID_IMAGE, "and", RESULTS_FILE)


if __name__ == "__main__":
    main()
