"""Day 6: a CNN that sorts product images into catalog categories.

Steps:
    1. Prototype on Fashion-MNIST (28x28 grey pictures of 10 kinds of clothing) to test the design quickly.
       We compare a plain dense network, a CNN without augmentation, and a CNN with augmentation.
    2. Look at the confusion matrix: which categories does the best model mix up?
    3. Save the catalog classifier (our first computer-vision module).
    4. If the real "Fashion Product Images" are on this computer, repeat the idea on them
       (loading pictures from disk + augmentation). If not, say how to get them.

Run from the main project folder:   python -m src.train_day6
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's chatty startup messages

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import (CNN_MODEL_FILE, DAY6_RESULTS_FILE, MODEL_FOLDER, PRODUCT_CNN_MODEL_FILE,
                        PRODUCT_IMAGES_FOLDER, RANDOM_STATE)

FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# ---------------------------------------------------------------- building blocks
def build_dense_network(input_shape, number_of_classes):
    """The Day 5 idea: flatten the picture into one long row of numbers, then dense layers."""
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_shape, number_of_classes, augment):
    """A small CNN: [convolution -> pooling] twice, then dropout, then dense layers.

    - Convolution layer: slides small 3x3 filters over the picture to find edges, then shapes.
    - Pooling layer: keeps the strongest signal in each 2x2 block, so the picture gets smaller.
    - Dropout: randomly switches off 40% of the neurons while training so it cannot memorise.
    - augment=True adds random flips, small turns and zooms. They only happen while TRAINING,
      so the network sees many slightly different versions of each picture.
    """
    layers = [keras.Input(shape=input_shape)]
    if augment:
        layers += [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.1),
        ]
    layers += [
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(number_of_classes, activation="softmax"),
    ]
    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, x_train, y_train):
    """Train with early stopping on a validation slice (10% of the training pictures)."""
    # Up to 40 rounds; stop when the validation loss has not improved for 5 rounds in a row.
    # (A lower cap would cut off the slower-learning augmented network before it has finished.)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=40, batch_size=128,
                        callbacks=[early_stop], verbose=0)
    validation_losses = history.history["val_loss"]
    best_round = int(np.argmin(validation_losses))
    return {
        "parameters": int(model.count_params()),
        "rounds_trained": len(validation_losses),
        "best_round": best_round + 1,
        "best_validation_loss": round(float(validation_losses[best_round]), 4),
        "best_validation_accuracy": round(float(history.history["val_accuracy"][best_round]), 4),
    }


def evaluate_classifier(model, x_test, y_test, class_names):
    """Accuracy, per-category recall and the confusion matrix on pictures the model has never seen."""
    probabilities = model.predict(x_test, verbose=0)
    guesses = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_test, guesses, labels=list(range(len(class_names))))

    # recall per category: of all real "Shirt" pictures, what share did we call "Shirt"?
    per_class = {}
    for number, name in enumerate(class_names):
        total = int(matrix[number].sum())
        per_class[name] = round(float(matrix[number, number] / total), 4) if total > 0 else None

    # the biggest mix-ups: real category -> guessed category (ignore the correct cells on the diagonal)
    mix_ups = []
    for real in range(len(class_names)):
        for guessed in range(len(class_names)):
            if real != guessed and matrix[real, guessed] > 0:
                mix_ups.append((int(matrix[real, guessed]), class_names[real], class_names[guessed]))
    mix_ups.sort(reverse=True)

    return {
        "accuracy": round(float((guesses == y_test).mean()), 4),
        "per_class_recall": per_class,
        "top_mix_ups": [{"times": t, "real": r, "guessed_as": g} for t, r, g in mix_ups[:5]],
        "confusion_matrix": matrix.tolist(),
    }


def print_confusion_matrix(matrix, class_names):
    """Print the confusion matrix: each row is the REAL category, each column is what we GUESSED."""
    short = [name[:5] for name in class_names]
    print("  " + " " * 12 + "".join(f"{s:>6}" for s in short))
    for number, name in enumerate(class_names):
        print(f"  {name:<12}" + "".join(f"{value:>6}" for value in matrix[number]))


# ---------------------------------------------------------------- Step 1: Fashion-MNIST
def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    # pixels are 0-255; scale to 0-1. Add a "channel" axis: a grey picture has 1 channel.
    x_train = (x_train / 255.0).astype("float32")[..., np.newaxis]
    x_test = (x_test / 255.0).astype("float32")[..., np.newaxis]
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------- Step 4: real product photos
def load_product_images(folder=PRODUCT_IMAGES_FOLDER, image_size=64, number_of_classes=10, per_class=400):
    """Load real product photos from disk.

    Expects the Kaggle "fashion-product-images-small" layout:
        <folder>/styles.csv     (a table with an 'id' column and an 'articleType' column)
        <folder>/images/<id>.jpg
    We keep the 10 most common article types and at most `per_class` photos of each, so the
    classes are balanced and it trains quickly on a laptop. Returns None if the data is not there.
    """
    styles_file = os.path.join(folder, "styles.csv")
    image_folder = os.path.join(folder, "images")
    if not (os.path.exists(styles_file) and os.path.isdir(image_folder)):
        return None

    # a few rows of the real styles.csv are broken, so skip bad lines
    styles = pd.read_csv(styles_file, on_bad_lines="skip")
    styles = styles[["id", "articleType"]].dropna()

    top_types = styles["articleType"].value_counts().head(number_of_classes).index.tolist()
    pictures = []
    labels = []
    for number, article_type in enumerate(top_types):
        rows = styles[styles["articleType"] == article_type]
        rows = rows.sample(n=min(per_class, len(rows)), random_state=RANDOM_STATE)
        for picture_id in rows["id"]:
            path = os.path.join(image_folder, f"{int(picture_id)}.jpg")
            if not os.path.exists(path):
                continue    # some ids in the table have no photo
            picture = keras.utils.load_img(path, target_size=(image_size, image_size))   # resize + colour
            pictures.append(keras.utils.img_to_array(picture) / 255.0)
            labels.append(number)

    return np.array(pictures, dtype="float32"), np.array(labels), top_types


def run_on_product_images():
    """Step 4: the same CNN design on real product photos. Returns the results, or None if no data."""
    loaded = load_product_images()
    if loaded is None:
        print("\nStep 4: real product photos not found in", PRODUCT_IMAGES_FOLDER)
        print("  To run this step: download 'fashion-product-images-small' from Kaggle")
        print("  (paramaggarwal/fashion-product-images-small), unzip it so that")
        print(f"  {PRODUCT_IMAGES_FOLDER}/styles.csv and {PRODUCT_IMAGES_FOLDER}/images/ exist, then run this again.")
        return None

    pictures, labels, class_names = loaded
    print(f"\nStep 4: real product photos: {len(pictures)} pictures, {len(class_names)} categories")
    x_train, x_test, y_train, y_test = train_test_split(
        pictures, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)

    keras.utils.set_random_seed(RANDOM_STATE)
    model = build_cnn(x_train.shape[1:], len(class_names), augment=True)
    training = train_model(model, x_train, y_train)
    scores = evaluate_classifier(model, x_test, y_test, class_names)
    print(f"  CNN with augmentation: test accuracy {scores['accuracy']}")
    print_confusion_matrix(scores["confusion_matrix"], class_names)

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    model.save(PRODUCT_CNN_MODEL_FILE)
    print("  Saved ->", PRODUCT_CNN_MODEL_FILE)
    return {"classes": class_names, "pictures": int(len(pictures)), "training": training, "test": scores}


# ---------------------------------------------------------------- main
def main():
    print("Step 1: prototype on Fashion-MNIST (28x28 pictures, 10 categories)")
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    print(f"  {len(x_train)} training pictures, {len(x_test)} test pictures")

    designs = {
        "Dense network (flattened pixels)": lambda: build_dense_network(x_train.shape[1:], 10),
        "CNN, no augmentation": lambda: build_cnn(x_train.shape[1:], 10, augment=False),
        "CNN with augmentation": lambda: build_cnn(x_train.shape[1:], 10, augment=True),
    }

    models = {}
    results = {}
    for name, make_model in designs.items():
        keras.utils.set_random_seed(RANDOM_STATE)      # the same starting point every run
        model = make_model()
        results[name] = {"training": train_model(model, x_train, y_train)}
        models[name] = model
        t = results[name]["training"]
        print(f"  {name}: {t['parameters']} weights, {t['rounds_trained']} rounds, "
              f"best validation loss {t['best_validation_loss']}")

    # choose by validation loss, never by the test pictures
    chosen = None
    for name in results:
        if chosen is None or (results[name]["training"]["best_validation_loss"]
                              < results[chosen]["training"]["best_validation_loss"]):
            chosen = name
    print("  lower validation loss wins ->", chosen)

    print("\nStep 2: test pictures")
    for name, model in models.items():
        results[name]["test"] = evaluate_classifier(model, x_test, y_test, FASHION_CLASSES)
        print(f"  {name:<34} test accuracy {results[name]['test']['accuracy']}")

    best = results[chosen]["test"]
    print(f"\n  Confusion matrix for {chosen} (rows = real, columns = guessed):")
    print_confusion_matrix(best["confusion_matrix"], FASHION_CLASSES)
    print("  Weakest categories (recall):",
          sorted(best["per_class_recall"].items(), key=lambda item: item[1])[:3])
    print("  Biggest mix-ups:")
    for mix in best["top_mix_ups"]:
        print(f"    {mix['real']} guessed as {mix['guessed_as']}: {mix['times']} times")

    dense_accuracy = results["Dense network (flattened pixels)"]["test"]["accuracy"]
    cnn_accuracy = results["CNN, no augmentation"]["test"]["accuracy"]
    aug_accuracy = results["CNN with augmentation"]["test"]["accuracy"]
    print(f"\n  CNN vs dense: {cnn_accuracy} vs {dense_accuracy} (difference {round(cnn_accuracy - dense_accuracy, 4)})")
    print(f"  augmentation: {aug_accuracy} with vs {cnn_accuracy} without (difference {round(aug_accuracy - cnn_accuracy, 4)})")

    print("\nStep 3: save the catalog classifier")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    models[chosen].save(CNN_MODEL_FILE)
    print("  Saved ->", CNN_MODEL_FILE)

    product_results = run_on_product_images()

    # the confusion matrix of the models we did not choose is not needed in the file
    summary = {name: {"training": r["training"],
                      "test_accuracy": r["test"]["accuracy"]} for name, r in results.items()}
    with open(DAY6_RESULTS_FILE, "w") as f:
        json.dump({
            "classes": FASHION_CLASSES,
            "models": summary,
            "chosen_model": chosen,
            "chosen_model_test": best,
            "cnn_minus_dense_accuracy": round(cnn_accuracy - dense_accuracy, 4),
            "augmentation_minus_none_accuracy": round(aug_accuracy - cnn_accuracy, 4),
            "product_images": product_results,
        }, f, indent=2)
    print("Saved results ->", DAY6_RESULTS_FILE)


if __name__ == "__main__":
    main()
