"""The RetailMind computer-vision module: classification + detection + segmentation + explanation.

One object, a few simple methods:

    from src.vision import CatalogVision
    vision = CatalogVision()

    result = vision.analyze(picture)              # grey 28x28 picture, values 0-255
    # {'category': 'Dress', 'confidence': 0.98, 'top_3': [...],
    #  'product_share': 0.41, 'heat_on_product': 0.93}
    # plus the pictures: result['mask'], result['clean'], result['heatmap'], result['explanation']

    cutout = vision.cut_out_from_photo("shoe.jpg")      # colour photo -> product on a white background

Classification is the Day 6 CNN, detection is the Day 7 YOLO detector (optional, it needs the
'ultralytics' package), segmentation is OpenCV (Day 8), explanation is Grad-CAM (Day 8).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np

from src.config import CNN_MODEL_FILE
from src.explain import grad_cam, heat_inside_mask, overlay
from src.segmentation import mask_area_share, put_on_background, segment_plain_background, segment_with_box
from src.train_day6 import FASHION_CLASSES


class CatalogVision:
    def __init__(self, model_file=CNN_MODEL_FILE):
        self.model = keras.models.load_model(model_file)

    def classify(self, picture):
        """picture: grey (28, 28) uint8. Returns (category, confidence, top_3, probabilities)."""
        ready = (picture / 255.0).astype("float32")[np.newaxis, ..., np.newaxis]
        probabilities = self.model.predict(ready, verbose=0)[0]
        best_three = probabilities.argsort()[::-1][:3]
        top_3 = [{FASHION_CLASSES[int(i)]: round(float(probabilities[i]), 4)} for i in best_three]
        return FASHION_CLASSES[int(best_three[0])], round(float(probabilities[best_three[0]]), 4), top_3

    def analyze(self, picture):
        """Run the whole pipeline on one grey 28x28 picture (values 0-255)."""
        category, confidence, top_3 = self.classify(picture)
        mask = segment_plain_background(picture)
        heatmap, _ = grad_cam(self.model, picture / 255.0)
        return {
            "category": category,
            "confidence": confidence,
            "top_3": top_3,
            "product_share": round(mask_area_share(mask), 3),            # how much of the picture is product
            "heat_on_product": round(heat_inside_mask(heatmap, mask), 3),  # is the model looking at the product?
            "mask": mask,
            "clean": put_on_background(picture, mask, 255),                # product on white
            "heatmap": heatmap,
            "explanation": overlay(picture, heatmap),
        }

    def detect(self, image_path):
        """Find objects and their boxes with the Day 7 YOLO detector (needs: pip install ultralytics)."""
        from src.detector import detect_products
        return detect_products(image_path)

    def cut_out_from_photo(self, image_path, box=None, background=(255, 255, 255)):
        """Colour photo -> the product on a uniform background.

        box = (x1, y1, x2, y2) around the product. If you do not give one, the most confident
        detection from the YOLO detector is used."""
        photo = cv2.imread(image_path)
        if photo is None:
            raise FileNotFoundError(f"Could not read the picture: {image_path}")
        if box is None:
            found = self.detect(image_path)
            if not found:
                raise ValueError("The detector found nothing to cut out. Pass a box=(x1, y1, x2, y2).")
            box = max(found, key=lambda item: item["confidence"])["box"]
        mask = segment_with_box(photo, box)
        return put_on_background(photo, mask, background), mask
