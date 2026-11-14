"""Day 11: make extra pictures of a thin product category with a VAE, and see if the classifier gets fairer.

The problem: some categories have few training pictures, and the classifier is weakest on them.
The idea: train a VAE (a network that learns to draw new pictures that look like its training pictures)
on the few real pictures of the thin category, then add its drawings to the classifier's training set.

Steps:
    1. Find the weakest category in the Day 6 results. Pretend we only have 100 training pictures of it.
    2. Train a VAE on those 100 pictures and draw 900 new ones.
    3. Inspect them: are they realistic, are they different from each other (mode collapse), are they copies?
       Keep only the realistic ones.
    4. Train the Day 6 CNN three ways, 5 times each (5 random starts): thin data, the 100 real pictures
       simply repeated (a fair control), and thin data + the VAE's pictures. Compare per-category recall.

Run from the main project folder (needs src.train_day6 first; about 15 minutes on a laptop CPU):
    python -m src.train_day11
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's chatty startup messages

import cv2
import keras
import numpy as np
import tensorflow as tf

from src.config import (DAY6_RESULTS_FILE, DAY11_RESULTS_FILE, MODEL_FOLDER, RANDOM_STATE,
                        SYNTHETIC_IMAGE, VAE_DECODER_FILE)
from src.train_day6 import FASHION_CLASSES, build_cnn, evaluate_classifier, load_fashion_mnist, train_model

KEEP_THIN = 100          # pretend we only have 100 pictures of the thin category
PER_CLASS = 1000         # and 1,000 of every other category (a smaller set keeps training quick)
NEW_PICTURES = 900       # how many pictures the VAE draws (100 real + 900 drawn = 1,000, like the others)
SEEDS = [1, 2, 3, 4, 5]


def weakest_category():
    """The category with the lowest recall in the Day 6 results: the one that needs help most."""
    with open(DAY6_RESULTS_FILE) as f:
        recall = json.load(f)["chosen_model_test"]["per_class_recall"]
    name = min(recall, key=recall.get)
    return FASHION_CLASSES.index(name), recall[name]


def make_thin_training_set(x, y, thin_class):
    """PER_CLASS pictures of every category, except the thin one, which gets only KEEP_THIN."""
    random = np.random.RandomState(RANDOM_STATE)
    chosen = []
    for category in range(10):
        how_many = KEEP_THIN if category == thin_class else PER_CLASS
        chosen.append(random.choice(np.where(y == category)[0], how_many, replace=False))
    chosen = np.concatenate(chosen)
    return x[chosen], y[chosen]


class VAE(keras.Model):
    """A small variational autoencoder for 28x28 pictures.

    - The encoder squeezes a picture into two short lists of numbers (mean and spread), a "latent code".
    - We pick a random point near that mean and the decoder draws a picture from it.
    - Training makes the drawing match the original (reconstruction loss) while keeping the codes close to a
      standard bell curve (KL loss). That second part is what lets us draw NEW pictures from random codes later.
    """

    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([keras.Input(shape=(784,)), keras.layers.Dense(256, activation="relu")])
        self.to_mean = keras.layers.Dense(latent_dim)
        self.to_logvar = keras.layers.Dense(latent_dim)
        self.decoder = keras.Sequential([keras.Input(shape=(latent_dim,)), keras.layers.Dense(256, activation="relu"),
                                         keras.layers.Dense(784, activation="sigmoid")])
        self.loss_tracker = keras.metrics.Mean(name="loss")      # lets fit() report the loss our train_step computes

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, pictures):
        with tf.GradientTape() as tape:
            hidden = self.encoder(pictures)
            mean, logvar = self.to_mean(hidden), self.to_logvar(hidden)
            code = mean + tf.exp(0.5 * logvar) * tf.random.normal(tf.shape(mean))     # a random point near the mean
            drawn = self.decoder(code)
            reconstruction = 784 * keras.losses.binary_crossentropy(pictures, drawn)  # how far the drawing is from the original
            kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
            loss = tf.reduce_mean(reconstruction + kl)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_weights), self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def draw_new_pictures(vae, how_many, seed):
    codes = np.random.RandomState(seed).normal(size=(how_many, vae.latent_dim)).astype("float32")
    return vae.decoder.predict(codes, verbose=0).reshape(-1, 28, 28, 1)


def nearest_distances(a, b, skip_self=False):
    """For every picture in a, the distance to its closest picture in b."""
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    distances = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    if skip_self:
        np.fill_diagonal(distances, np.inf)          # a picture is not its own neighbour
    return distances.min(axis=1)


def inspect_pictures(real, drawn):
    """Quality checks on the VAE's pictures. Returns the numbers and which pictures to keep."""
    real_to_real = nearest_distances(real, real, skip_self=True)       # how close real pictures are to each other
    drawn_to_real = nearest_distances(drawn, real)                     # how close drawn pictures are to real ones
    limit = float(np.percentile(real_to_real, 95))                     # drawn pictures farther than this look unlike any real one
    keep = drawn_to_real <= limit

    flat_real, flat_drawn = real.reshape(len(real), -1), drawn.reshape(len(drawn), -1)
    sample = flat_drawn[:200]
    spread_drawn = float(np.sqrt(((sample[:, None] - sample[None]) ** 2).sum(axis=2)).mean())     # mode collapse check:
    spread_real = float(np.sqrt(((flat_real[:, None] - flat_real[None]) ** 2).sum(axis=2)).mean())  # are they as varied as real?
    report = {
        "real_to_real_nearest_mean": round(float(real_to_real.mean()), 3),
        "drawn_to_real_nearest_mean": round(float(drawn_to_real.mean()), 3),
        "limit_used_to_keep": round(limit, 3),
        "drawn_kept": int(keep.sum()), "drawn_total": int(len(drawn)),
        "variety_between_real": round(spread_real, 3), "variety_between_drawn": round(spread_drawn, 3),
        "mean_brightness_real": round(float(real.mean()), 4), "mean_brightness_drawn": round(float(drawn.mean()), 4),
    }
    return report, keep


def save_picture_grid(real, drawn, path):
    """Top 2 rows: real pictures. Next 4 rows: the VAE's pictures. Each enlarged 4x so it is readable."""
    def row(pictures):
        tiles = [cv2.resize((p[:, :, 0] * 255).astype("uint8"), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST) for p in pictures]
        return np.hstack(tiles)
    rows = [row(real[:12]), row(real[12:24]), np.full((8, 12 * 112), 255, np.uint8)]
    rows += [row(drawn[i * 12:(i + 1) * 12]) for i in range(4)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack(rows))


def train_and_score(x_train, y_train, x_test, y_test, seed):
    """Train the Day 6 CNN once from a given random start; return its per-category recall and accuracy."""
    shuffle = np.random.RandomState(0).permutation(len(x_train))      # shuffle so the validation slice mixes categories
    keras.utils.set_random_seed(seed)
    model = build_cnn(x_train.shape[1:], 10, augment=False)
    train_model(model, x_train[shuffle], y_train[shuffle])
    scores = evaluate_classifier(model, x_test, y_test, FASHION_CLASSES)
    return scores["per_class_recall"], scores["accuracy"]


def summarise(runs, thin_name):
    """Average over the random starts: accuracy, thin-category recall, the other categories, and the spread."""
    accuracy = np.array([a for _, a in runs])
    thin = np.array([r[thin_name] for r, _ in runs])
    others = np.array([np.mean([v for k, v in r.items() if k != thin_name]) for r, _ in runs])
    return {"accuracy_mean": round(float(accuracy.mean()), 4), "accuracy_std": round(float(accuracy.std()), 4),
            "thin_recall_mean": round(float(thin.mean()), 4), "thin_recall_std": round(float(thin.std()), 4),
            "thin_recall_per_seed": [round(float(v), 3) for v in thin],
            "other_categories_recall_mean": round(float(others.mean()), 4),
            "tshirt_recall_mean": round(float(np.mean([r["T-shirt/top"] for r, _ in runs])), 4)}


def main():
    thin_class, day6_recall = weakest_category()
    thin_name = FASHION_CLASSES[thin_class]
    print(f"Step 1: the weakest category on Day 6 is '{thin_name}' (recall {day6_recall}). Pretend we have only {KEEP_THIN} pictures of it.")

    x_all, y_all, x_test, y_test = load_fashion_mnist()
    x_train, y_train = make_thin_training_set(x_all, y_all, thin_class)
    real = x_train[y_train == thin_class]
    print(f"  training set: {len(x_train)} pictures ({len(real)} of '{thin_name}'); test set: {len(x_test)} pictures")

    print("\nStep 2: train a VAE on the thin category and draw new pictures")
    keras.utils.set_random_seed(RANDOM_STATE)
    vae = VAE()
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    history = vae.fit(real.reshape(len(real), 784), epochs=300, batch_size=20, verbose=0)
    print(f"  loss went from {history.history['loss'][0]:.1f} to {history.history['loss'][-1]:.1f}")
    drawn = draw_new_pictures(vae, NEW_PICTURES, RANDOM_STATE)

    print("\nStep 3: inspect the drawn pictures")
    report, keep = inspect_pictures(real, drawn)
    for name, value in report.items():
        print(f"  {name}: {value}")
    kept = drawn[keep]
    save_picture_grid(real, kept if len(kept) >= 48 else drawn, SYNTHETIC_IMAGE)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    vae.decoder.save(VAE_DECODER_FILE)

    print("\nStep 4: train the CNN three ways from 5 random starts each (this is the slow part)")
    repeated = real[np.arange(NEW_PICTURES) % len(real)]               # the control: just repeat the real pictures
    setups = {
        "thin data only": (x_train, y_train),
        "real pictures repeated": (np.concatenate([x_train, repeated]),
                                   np.concatenate([y_train, np.full(NEW_PICTURES, thin_class)])),
        "thin data + VAE pictures": (np.concatenate([x_train, kept]),
                                     np.concatenate([y_train, np.full(len(kept), thin_class)])),
    }
    results = {}
    for name, (x_set, y_set) in setups.items():
        runs = []
        for seed in SEEDS:
            runs.append(train_and_score(x_set, y_set, x_test, y_test, seed))
            print(f"  {name}, start {seed}: {thin_name} recall {runs[-1][0][thin_name]}, accuracy {runs[-1][1]}", flush=True)
        results[name] = summarise(runs, thin_name)

    print(f"\n  {'Setup':<28}{thin_name + ' recall':>16}{'accuracy':>11}{'other cats':>12}{'T-shirt':>9}")
    for name, s in results.items():
        print(f"  {name:<28}{s['thin_recall_mean']:>9} (+-{s['thin_recall_std']:.3f}){s['accuracy_mean']:>9}{s['other_categories_recall_mean']:>12}{s['tshirt_recall_mean']:>9}")

    with open(DAY11_RESULTS_FILE, "w") as f:
        json.dump({"thin_category": thin_name, "day6_recall": day6_recall, "real_pictures_kept": KEEP_THIN,
                   "other_pictures_per_category": PER_CLASS, "drawn_total": int(len(drawn)), "inspection": report,
                   "seeds": SEEDS, "setups": results}, f, indent=2)
    print("\nSaved ->", DAY11_RESULTS_FILE, ",", SYNTHETIC_IMAGE, "and", VAE_DECODER_FILE)


if __name__ == "__main__":
    main()
