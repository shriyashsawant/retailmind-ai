"""Day 5: a Keras neural network (ANN) for churn, compared head-to-head with the best classical model.

Steps:
    1. Warm-up: a tiny network on the MNIST handwritten digits (to see layers, activations, training).
    2. Build two churn networks (a shallow one and a deeper one) and train them with early stopping.
    3. Pick the better network using validation loss (not the test customers).
    4. Test it against the best classical model from Phase 1 and say honestly whether depth helped.
    5. Save the network, the data-preparation step and the results.

Run from the main project folder:   python -m src.train_day5
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's chatty startup messages

import joblib
import keras
import numpy as np

from src.config import (ANN_MODEL_FILE, ANN_PREPARE_FILE, BEST_MODEL_FILE, DAY5_RESULTS_FILE,
                        MODEL_FOLDER, RANDOM_STATE)
from src.data import get_train_test_data
from src.evaluate import evaluate, evaluate_probabilities
from src.pipeline import make_preprocessor


def mnist_warm_up():
    """Train a tiny network on handwritten digits. Only here to see how a Keras network works."""
    print("Step 1: warm-up on MNIST (images of the digits 0 to 9)")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as problem:     # for example no internet the first time
        print("  skipped (could not load MNIST):", problem)
        return None

    # each picture is 28x28 pixels with values 0-255. Flatten to 784 numbers and scale to 0-1.
    x_train = x_train[:10000].reshape(-1, 784) / 255.0
    y_train = y_train[:10000]
    x_test = x_test[:2000].reshape(-1, 784) / 255.0
    y_test = y_test[:2000]

    keras.utils.set_random_seed(RANDOM_STATE)    # the same starting point every run
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        keras.layers.Dense(64, activation="relu"),        # hidden layer: 64 neurons
        keras.layers.Dense(10, activation="softmax"),     # output layer: one probability per digit
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  after 3 rounds (epochs) the network gets {accuracy:.1%} of unseen digits right")
    return round(float(accuracy), 4)


def build_network(number_of_inputs, hidden_layers, dropout=0.0):
    """A feed-forward network: inputs -> hidden layers -> one sigmoid output (the churn probability)."""
    layers = [keras.Input(shape=(number_of_inputs,))]
    for neurons in hidden_layers:
        layers.append(keras.layers.Dense(neurons, activation="relu"))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))   # randomly switches neurons off so it cannot memorise
    layers.append(keras.layers.Dense(1, activation="sigmoid"))

    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy")
    return model


def main():
    mnist_accuracy = mnist_warm_up()

    # ---- the churn data: same customers as every other day, original columns ----
    print("\nStep 2: build and train the churn networks")
    X_train, X_test, y_train, y_test = get_train_test_data()

    # Networks need numbers on a similar scale, so we reuse our scaling + one-hot step.
    # It is learned from the training customers only (no peeking at the test ones).
    prepare = make_preprocessor(X_train)
    X_train_ready = prepare.fit_transform(X_train).astype("float32")
    X_test_ready = prepare.transform(X_test).astype("float32")
    number_of_inputs = X_train_ready.shape[1]
    print("  input columns after preparation:", number_of_inputs)

    # Only about 1 in 4 customers churns. Like class_weight="balanced" on Day 2, this makes the
    # network count a churner more heavily so it does not just guess "stays" every time.
    stayers = int((y_train == 0).sum())
    churners = int((y_train == 1).sum())
    class_weight = {0: 1.0, 1: round(stayers / churners, 2)}

    designs = {
        "Shallow ANN (1 hidden layer of 16)": dict(hidden_layers=[16], dropout=0.0),
        "Deeper ANN (3 hidden layers: 64, 32, 16)": dict(hidden_layers=[64, 32, 16], dropout=0.3),
    }

    networks = {}
    results = {}
    for name, design in designs.items():
        keras.utils.set_random_seed(RANDOM_STATE)    # the same starting point every run
        network = build_network(number_of_inputs, **design)

        # Early stopping: watch the loss on a validation slice (20% of the training customers) and
        # stop when it has not improved for 10 rounds. restore_best_weights goes back to the best round.
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)
        history = network.fit(
            X_train_ready, y_train.to_numpy(), validation_split=0.2, epochs=100, batch_size=32,
            class_weight=class_weight, callbacks=[early_stop], verbose=0)

        validation_losses = history.history["val_loss"]
        best_round = int(np.argmin(validation_losses)) + 1
        parameters = int(network.count_params())
        print(f"  {name}: {parameters} weights, stopped after {len(validation_losses)} rounds, "
              f"best round {best_round}, best validation loss {min(validation_losses):.4f}")

        networks[name] = network
        results[name] = {
            "parameters": parameters,
            "rounds_trained": len(validation_losses),
            "best_round": best_round,
            "best_validation_loss": round(float(min(validation_losses)), 4),
            "final_training_loss": round(float(history.history["loss"][-1]), 4),
        }

    # ---- Step 3: pick the better network by validation loss (never by the test customers) ----
    print("\nStep 3: choose the network")
    chosen = None
    for name in results:
        if chosen is None or results[name]["best_validation_loss"] < results[chosen]["best_validation_loss"]:
            chosen = name
    print("  lower validation loss wins ->", chosen)

    # ---- Step 4: test everything on the test customers (once) ----
    print("\nStep 4: test customers, ANN vs the best classical model")
    for name, network in networks.items():
        probabilities = network.predict(X_test_ready, verbose=0).ravel()
        results[name]["test_scores"] = evaluate_probabilities(probabilities, y_test)

    classical = joblib.load(BEST_MODEL_FILE)
    classical_scores = evaluate(classical, X_test, y_test)

    table = {"Best classical model (Bagging, Phase 1)": classical_scores}
    for name in networks:
        table[name] = results[name]["test_scores"]

    print(f"  {'Model':<44}{'Recall':>8}{'Precision':>11}{'F1':>8}{'ROC-AUC':>9}")
    for name, s in table.items():
        mark = "  <- chosen ANN" if name == chosen else ""
        print(f"  {name:<44}{s['recall']:>8}{s['precision']:>11}{s['f1']:>8}{s['roc_auc']:>9}{mark}")

    ann = results[chosen]["test_scores"]
    f1_gap = round(ann["f1"] - classical_scores["f1"], 4)
    auc_gap = round(ann["roc_auc"] - classical_scores["roc_auc"], 4)
    if f1_gap > 0.01:
        verdict = f"Depth helped: the ANN F1 is {f1_gap} higher than the classical model."
    elif f1_gap < -0.01:
        verdict = f"Depth did not help: the ANN F1 is {abs(f1_gap)} lower than the classical model."
    else:
        verdict = (f"Depth did not clearly help: the ANN F1 differs from the classical model by {f1_gap} "
                   f"(ROC-AUC by {auc_gap}), which is within normal test noise.")

    shallow = results["Shallow ANN (1 hidden layer of 16)"]["test_scores"]["f1"]
    deep = results["Deeper ANN (3 hidden layers: 64, 32, 16)"]["test_scores"]["f1"]
    print("  ->", verdict)
    print(f"  -> shallow vs deeper network: F1 {shallow} vs {deep}")

    # ---- Step 5: save ----
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    networks[chosen].save(ANN_MODEL_FILE)
    joblib.dump(prepare, ANN_PREPARE_FILE)
    with open(DAY5_RESULTS_FILE, "w") as f:
        json.dump({
            "mnist_warm_up_accuracy": mnist_accuracy,
            "class_weight": class_weight,
            "networks": results,
            "chosen_network": chosen,
            "classical_champion_test_scores": classical_scores,
            "f1_gap_vs_classical": f1_gap,
            "roc_auc_gap_vs_classical": auc_gap,
            "verdict": verdict,
        }, f, indent=2)
    print("\nSaved network     ->", ANN_MODEL_FILE)
    print("Saved preparation ->", ANN_PREPARE_FILE)
    print("Saved results     ->", DAY5_RESULTS_FILE)


if __name__ == "__main__":
    main()
