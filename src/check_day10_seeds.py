"""Day 10 check: is "attention did not help" real, or just one unlucky random start?

A neural network starts from random numbers, so two runs of the same code give slightly different scores.
Here we train the Seq2Seq model WITH and WITHOUT attention from 5 different random starts and compare the
test error. If the gap between them is smaller than the run-to-run spread, it is noise.

Run from the main project folder (about 10 minutes):   python -m src.check_day10_seeds
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from src.demand_data import daily_table, load_sales, mean_absolute_error, scale_windows, split_windows
from src.train_day10 import HORIZON, LOOKBACK, build_seq2seq, train

SEEDS = [1, 2, 3, 4, 5]
RESULTS_FILE = "models/day10_seed_check.json"


def main():
    table, source = load_sales()
    train_set, val_set, test_set = split_windows(daily_table(table), LOOKBACK, HORIZON)
    x_train, y_train, _, _ = scale_windows(*train_set[:2])
    x_val, y_val, _, _ = scale_windows(*val_set[:2])
    x_test, _, test_average, test_spread = scale_windows(test_set[0])
    real = test_set[1]

    errors = {"without attention": [], "with attention": []}
    for seed in SEEDS:
        for name, use_attention in [("without attention", False), ("with attention", True)]:
            keras.utils.set_random_seed(seed)
            model, _ = build_seq2seq(use_attention)
            train(model, x_train, y_train, x_val, y_val)
            guess = np.maximum(model.predict(x_test[..., np.newaxis], verbose=0) * test_spread + test_average, 0)
            errors[name].append(round(mean_absolute_error(real, guess), 1))
        print(f"  seed {seed}: without attention {errors['without attention'][-1]} | with attention {errors['with attention'][-1]}", flush=True)

    summary = {name: {"mae_per_seed": values, "mean": round(float(np.mean(values)), 1), "std": round(float(np.std(values)), 1)}
              for name, values in errors.items()}
    wins = sum(a < b for a, b in zip(errors["with attention"], errors["without attention"]))
    gap = round(summary["with attention"]["mean"] - summary["without attention"]["mean"], 1)
    print("\nMAE over", len(SEEDS), "random starts (lower is better)")
    for name, s in summary.items():
        print(f"  {name:<20} mean {s['mean']}  spread (std) {s['std']}")
    print(f"  attention is better in {wins} of {len(SEEDS)} runs; mean gap (with minus without) = {gap}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"data": source, "seeds": SEEDS, "summary": summary,
                   "attention_better_in_runs": wins, "mean_gap_with_minus_without": gap}, f, indent=2)
    print("  Saved ->", RESULTS_FILE)


if __name__ == "__main__":
    main()
