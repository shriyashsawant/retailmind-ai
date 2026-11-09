"""Find and locate objects in a picture with a pretrained YOLO model (Ultralytics).

Classification (Day 6) answers "what is this picture?". Detection answers "what is in this picture,
and where?" It draws a box around each object. That helps with catalog quality checks, pictures
with several products in them, and shelf analytics.

The pretrained model learned on the COCO photos (80 everyday object types). It was NOT trained on
fashion catalog pictures, so treat it as a general-purpose finder, not a catalog expert.

Needs:   pip install ultralytics      (this also installs PyTorch)

Run from the main project folder:
    python -m src.detector path/to/picture.jpg
    python -m src.detector path/to/picture.jpg --save reports/boxes.jpg
"""
import argparse
import json


def load_detector(weights="yolov8n.pt"):
    """Load a pretrained YOLO model. 'yolov8n.pt' is the smallest one (about 6 MB, downloaded the first time)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Detection needs Ultralytics. Install it with:  pip install ultralytics")
    return YOLO(weights)


def detect_products(image_path, model=None, confidence=0.25):
    """Return one dictionary per object found: its label, how sure the model is, and its box."""
    model = model or load_detector()
    result = model.predict(image_path, conf=confidence, verbose=False)[0]

    found = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(float(value), 1) for value in box.xyxy[0]]   # corners of the box, in pixels
        found.append({
            "label": result.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "box": [x1, y1, x2, y2],
        })
    return found


def save_annotated(image_path, out_path, model=None, confidence=0.25):
    """Save a copy of the picture with the boxes and labels drawn on it."""
    from PIL import Image
    model = model or load_detector()
    result = model.predict(image_path, conf=confidence, verbose=False)[0]
    drawn = result.plot()                                    # a picture as an array, in BGR colour order
    Image.fromarray(drawn[:, :, ::-1]).save(out_path)        # flip BGR -> RGB, then save
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in a picture with YOLO.")
    parser.add_argument("image")
    parser.add_argument("--save", help="also save a copy with the boxes drawn on it")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    detector = load_detector()
    objects = detect_products(args.image, detector, args.confidence)
    print(json.dumps(objects, indent=2))
    if args.save:
        save_annotated(args.image, args.save, detector, args.confidence)
        print("Saved ->", args.save)
