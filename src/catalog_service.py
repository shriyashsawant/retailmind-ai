"""Tag catalog pictures with the Day 6 image classifier (the first computer-vision module).

classify_images() takes grey 28x28 pictures (like Fashion-MNIST) and returns the category the
model thinks each one shows, with how sure it is. Run from the main project folder
(after src.train_day6):

    python -m src.catalog_service        # tags 8 sample pictures and shows the real answer too
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from src.config import CNN_MODEL_FILE
from src.train_day6 import FASHION_CLASSES, load_fashion_mnist


def classify_images(pictures):
    """pictures: an array of shape (number, 28, 28), pixel values 0-255.

    Returns one answer per picture: the category, the confidence and the top 3 guesses.
    """
    model = keras.models.load_model(CNN_MODEL_FILE)
    ready = (np.asarray(pictures) / 255.0).astype("float32")[..., np.newaxis]   # same scaling as training
    probabilities = model.predict(ready, verbose=0)

    answers = []
    for number, row in enumerate(probabilities, start=1):
        best_three = row.argsort()[::-1][:3]
        answers.append({
            "picture": number,
            "category": FASHION_CLASSES[int(best_three[0])],
            "confidence": round(float(row[best_three[0]]), 4),
            "top_3": [{FASHION_CLASSES[int(i)]: round(float(row[i]), 4)} for i in best_three],
        })
    return answers


if __name__ == "__main__":
    _, _, x_test, y_test = load_fashion_mnist()
    samples = (x_test[:8, :, :, 0] * 255).astype("uint8")      # back to 0-255 pictures
    for answer, real in zip(classify_images(samples), y_test[:8]):
        answer["real_category"] = FASHION_CLASSES[int(real)]
        print(json.dumps(answer))
