"""Day 9 check: is the LSTM really better than the simple baselines, or did one random start flatter it?

A neural network starts from random numbers, so two runs of the same code give slightly different scores.
Here we train the Day 9 LSTM from 5 different random starts and compare each one with the best simple
baseline ("same weekday 2 weeks ago", which does not learn anything and so does not change between runs).

Run from the main project folder (about 5 minutes):   python -m src.check_day9_seeds
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from src.demand_data import daily_table, load_sales, mean_absolute_error, scale_windows, split_windows
from src.train_day9 import HORIZON, LOOKBACK, build_lstm

SEEDS = [1, 2, 3, 4, 5]
RESULTS_FILE = "models/day9_seed_check.json"


def main():
    table, source = load_sales()
    train_set, val_set, test_set = split_windows(daily_table(table), LOOKBACK, HORIZON)
    x_train, y_train, _, _ = scale_windows(*train_set[:2])
    x_val, y_val, _, _ = scale_windows(*val_set[:2])
    x_test, _, test_average, test_spread = scale_windows(test_set[0])
    real = test_set[1][:, 0]
    baseline = round(mean_absolute_error(real, test_set[0][:, -14]), 1)        # same weekday, 2 weeks earlier

    errors = []
    for seed in SEEDS:
        keras.utils.set_random_seed(seed)
        model = build_lstm()
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(x_train[..., np.newaxis], y_train[:, 0], validation_data=(x_val[..., np.newaxis], y_val[:, 0]),
                  epochs=150, batch_size=64, callbacks=[early_stop], verbose=0)
        guess = np.maximum(model.predict(x_test[..., np.newaxis], verbose=0)[:, 0] * test_spread[:, 0] + test_average[:, 0], 0)
        errors.append(round(mean_absolute_error(real, guess), 1))
        print(f"  seed {seed}: LSTM MAE {errors[-1]}", flush=True)

    beats = sum(e < baseline for e in errors)
    summary = {"lstm_mae_per_seed": errors, "mean": round(float(np.mean(errors)), 1), "std": round(float(np.std(errors)), 1),
               "baseline_same_weekday_2_weeks_ago": baseline, "runs_that_beat_baseline": beats}
    print(f"\nLSTM MAE over {len(SEEDS)} random starts: mean {summary['mean']}, spread (std) {summary['std']}")
    print(f"Best simple baseline (same weekday 2 weeks ago): {baseline}; the LSTM beats it in {beats} of {len(SEEDS)} runs")
    with open(RESULTS_FILE, "w") as f:
        json.dump({"data": source, "seeds": SEEDS, **summary}, f, indent=2)
    print("  Saved ->", RESULTS_FILE)


if __name__ == "__main__":
    main()
