"""Day 7: transfer learning. Does a pretrained ResNet50 beat the Day 6 CNN that learned from scratch?

Both models get the SAME training pictures (a 10,000-picture subset of Fashion-MNIST) and are tested
on the SAME 10,000 test pictures, so the comparison is fair.

Steps:
    1. Baseline: the Day 6 CNN, trained from scratch on the 10,000 pictures.
    2. Feature extraction: ResNet50 (learned on ImageNet) is frozen. We train only a new small "head" on top.
    3. Fine-tuning: unfreeze the top of ResNet50 and train it gently with a small learning rate.
    4. Compare all three and save the best model.

Run from the main project folder (needs TensorFlow; downloads the ResNet50 weights, about 100 MB, once):
    python -m src.train_day7
"""
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's chatty startup messages

import keras
import numpy as np

from src.config import DAY7_RESULTS_FILE, MODEL_FOLDER, RANDOM_STATE, RESNET_MODEL_FILE
from src.train_day6 import FASHION_CLASSES, build_cnn, evaluate_classifier, load_fashion_mnist, train_model

TRAIN_PICTURES = 10000       # a subset keeps training fast on a laptop CPU
INPUT_SIZE = 64              # ResNet50 wants bigger pictures than 28x28, so we enlarge them to 64x64
FINE_TUNE_LAYERS = 20        # how many layers at the top of ResNet50 we unfreeze


def pick_subset(x, y, how_many):
    """Take a random, reproducible subset of the pictures."""
    chosen = np.random.RandomState(RANDOM_STATE).choice(len(x), how_many, replace=False)
    return x[chosen], y[chosen]


def build_resnet_model(number_of_classes=10):
    """ResNet50 (pretrained on ImageNet) + a new head. The ResNet part starts out frozen."""
    backbone = keras.applications.ResNet50(include_top=False, weights="imagenet",
                                           input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    backbone.trainable = False                               # frozen: feature extraction only

    pictures = keras.Input(shape=(28, 28, 1))                # our grey 28x28 pictures, values 0 to 1
    x = keras.layers.Rescaling(255.0)(pictures)              # ResNet expects values 0 to 255
    x = keras.layers.Resizing(INPUT_SIZE, INPUT_SIZE)(x)     # enlarge to 64x64
    x = keras.layers.Concatenate()([x, x, x])                # ResNet expects 3 colour channels: copy the grey one 3 times
    x = keras.applications.resnet50.preprocess_input(x)      # ResNet's own preparation step
    x = backbone(x, training=False)                          # features from the frozen backbone
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    answers = keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = keras.Model(pictures, answers)
    return model, backbone


def fit_with_early_stopping(model, x_train, y_train, epochs):
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=64,
                        callbacks=[early_stop], verbose=2)
    return len(history.history["val_loss"])


def main():
    x_all, y_all, x_test, y_test = load_fashion_mnist()
    x_train, y_train = pick_subset(x_all, y_all, TRAIN_PICTURES)
    print(f"Training on {len(x_train)} pictures, testing on {len(x_test)} pictures")

    results = {"train_pictures": TRAIN_PICTURES, "test_pictures": int(len(x_test)), "models": {}}

    # ---- 1. baseline: the Day 6 CNN on the same small training set ----
    print("\nStep 1: the Day 6 CNN, learning from scratch")
    keras.utils.set_random_seed(RANDOM_STATE)
    started = time.time()
    baseline = build_cnn(x_train.shape[1:], 10, augment=False)
    training = train_model(baseline, x_train, y_train)
    scores = evaluate_classifier(baseline, x_test, y_test, FASHION_CLASSES)
    results["models"]["Day 6 CNN (from scratch)"] = {
        "weights": training["parameters"], "trainable_weights": training["parameters"],
        "test_accuracy": scores["accuracy"], "seconds": round(time.time() - started)}
    print("  test accuracy:", scores["accuracy"])

    # ---- 2. feature extraction: only the new head learns ----
    print("\nStep 2: ResNet50 frozen, train only the new head")
    keras.utils.set_random_seed(RANDOM_STATE)
    started = time.time()
    model, backbone = build_resnet_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print("  trainable weights (head only):", trainable, "of", model.count_params())
    fit_with_early_stopping(model, x_train, y_train, epochs=6)
    scores = evaluate_classifier(model, x_test, y_test, FASHION_CLASSES)
    results["models"]["ResNet50, frozen (feature extraction)"] = {
        "weights": int(model.count_params()), "trainable_weights": trainable,
        "test_accuracy": scores["accuracy"], "seconds": round(time.time() - started)}
    print("  test accuracy:", scores["accuracy"])

    # ---- 3. fine-tuning: unfreeze the top of ResNet50 and train it gently ----
    print(f"\nStep 3: fine-tune the top {FINE_TUNE_LAYERS} layers of ResNet50")
    started = time.time()
    backbone.trainable = True
    for layer in backbone.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False                              # the lower, general-purpose layers stay frozen
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False                          # keep the normalisation statistics steady
    model.compile(optimizer=keras.optimizers.Adam(5e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print("  trainable weights now:", trainable)
    fit_with_early_stopping(model, x_train, y_train, epochs=5)
    scores = evaluate_classifier(model, x_test, y_test, FASHION_CLASSES)
    fine_tuned = {
        "weights": int(model.count_params()), "trainable_weights": trainable,
        "test_accuracy": scores["accuracy"], "seconds": round(time.time() - started)}
    results["models"][f"ResNet50, fine-tuned (top {FINE_TUNE_LAYERS} layers)"] = fine_tuned
    results["fine_tuned_per_class_recall"] = scores["per_class_recall"]
    results["fine_tuned_top_mix_ups"] = scores["top_mix_ups"]
    print("  test accuracy:", scores["accuracy"])

    # ---- 4. compare and save ----
    day6_full_data = 0.9163                                   # the Day 6 CNN trained on all 60,000 pictures (see README)
    results["day6_cnn_on_all_60000_pictures"] = day6_full_data
    best_name = max(results["models"], key=lambda name: results["models"][name]["test_accuracy"])
    results["best_model"] = best_name

    print("\nStep 4: comparison on the same 10,000 test pictures")
    print(f"  {'Model':<46}{'Accuracy':>10}{'Seconds':>9}")
    for name, info in results["models"].items():
        print(f"  {name:<46}{info['test_accuracy']:>10}{info['seconds']:>9}")
    print(f"  (the Day 6 CNN trained on all 60,000 pictures scored {day6_full_data})")
    print("  best:", best_name)

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    model.save(RESNET_MODEL_FILE)
    with open(DAY7_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved model   ->", RESNET_MODEL_FILE, "(large file, not committed)")
    print("Saved results ->", DAY7_RESULTS_FILE)


if __name__ == "__main__":
    main()
