"""Day 9: an LSTM that forecasts tomorrow's sales for each store.

Steps:
    1. Load the daily sales and cut them into windows: the last 28 days -> the next day.
    2. Split by time: oldest days to train, then 8 weeks to validate, the latest 8 weeks to test.
    3. Set the bar with two simple baselines: "same as yesterday" and "same as this day last week".
    4. Train an LSTM, then score it on the test weeks with MAE and RMSE.
    5. Save the forecaster for the demand module (src/demand_service.py).

Run from the main project folder:   python -m src.train_day9
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's chatty startup messages

import keras
import numpy as np

from src.config import DAY9_RESULTS_FILE, DEMAND_LSTM_FILE, MODEL_FOLDER, RANDOM_STATE
from src.demand_data import (daily_table, load_sales, mean_absolute_error, root_mean_squared_error,
                             scale_windows, split_windows)

LOOKBACK = 28      # the model reads the last 28 days (4 full weeks)
HORIZON = 1        # and predicts the next day


def build_lstm():
    """28 days in -> one LSTM layer -> 1 number out (tomorrow's sales, in scaled units).

    The LSTM reads the days in order and keeps a memory of what it has seen, so it can pick up
    patterns like "Saturdays are busy" or "sales rise during a promotion".
    """
    model = keras.Sequential([
        keras.Input(shape=(LOOKBACK, 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def score(real, guess):
    return {"mae": round(mean_absolute_error(real, guess), 1), "rmse": round(root_mean_squared_error(real, guess), 1)}


def main():
    print("Step 1: load the sales")
    table, source = load_sales()
    wide = daily_table(table)
    print(f"  {source}")
    print(f"  {wide.shape[1]} stores x {wide.shape[0]} days ({wide.index[0].date()} to {wide.index[-1].date()})")

    print("\nStep 2: windows (28 days -> next day), split by time")
    train, validation, test = split_windows(wide, LOOKBACK, HORIZON)
    print(f"  train {len(train[0])} | validation {len(validation[0])} | test {len(test[0])} windows")

    x_train, y_train, _, _ = scale_windows(*train[:2])
    x_val, y_val, _, _ = scale_windows(*validation[:2])
    x_test, _, test_average, test_spread = scale_windows(test[0])
    real = test[1][:, 0]                                  # real sales on each test day

    print("\nStep 3: baselines (no learning)")
    yesterday = test[0][:, -1]                            # the last day of each window
    last_week = test[0][:, -7]                            # the same weekday, one week earlier
    two_weeks = test[0][:, -14]                           # the same weekday, two weeks earlier (promotions repeat every 2 weeks here)
    baselines = {"same as yesterday": score(real, yesterday), "same as last week": score(real, last_week),
                 "same as 2 weeks ago": score(real, two_weeks)}
    for name, s in baselines.items():
        print(f"  {name:<20} MAE {s['mae']:>8}  RMSE {s['rmse']:>8}")

    print("\nStep 4: train the LSTM")
    keras.utils.set_random_seed(RANDOM_STATE)             # the same starting point every run
    model = build_lstm()
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(x_train[..., np.newaxis], y_train[:, 0], validation_data=(x_val[..., np.newaxis], y_val[:, 0]),
                        epochs=150, batch_size=64, callbacks=[early_stop], verbose=0)   # a high cap: early stopping decides
    rounds = len(history.history["val_loss"])
    print(f"  {model.count_params()} weights, stopped after {rounds} rounds, "
          f"best validation loss {min(history.history['val_loss']):.4f}")

    scaled_guess = model.predict(x_test[..., np.newaxis], verbose=0)[:, 0]
    guess = np.maximum(scaled_guess * test_spread[:, 0] + test_average[:, 0], 0)   # back to real sales, never below 0
    lstm = score(real, guess)
    print(f"  LSTM on the test weeks:  MAE {lstm['mae']}  RMSE {lstm['rmse']}")

    best_baseline = min(baselines, key=lambda name: baselines[name]["mae"])
    gain = round(baselines[best_baseline]["mae"] - lstm["mae"], 1)
    if gain > 0:
        verdict = f"The LSTM beats the best baseline ({best_baseline}) by {gain} sales units of MAE."
    else:
        verdict = f"The LSTM does NOT beat the best baseline ({best_baseline}); it is {abs(gain)} units of MAE worse."
    print("  ->", verdict)

    # the first store's last 14 test days: a quick look at predicted vs actual (no plotting library needed)
    first_store = test[2] == test[2][0]
    sample = [{"actual": int(a), "lstm": int(g), "same_as_last_week": int(b)}
              for a, g, b in zip(real[first_store][-14:], guess[first_store][-14:], last_week[first_store][-14:])]
    print("\n  Store", test[2][0], "last 14 test days (actual | LSTM | same as last week)")
    for row in sample:
        print(f"    {row['actual']:>7} | {row['lstm']:>7} | {row['same_as_last_week']:>7}")

    print("\nStep 5: save the forecaster")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    model.save(DEMAND_LSTM_FILE)
    with open(DAY9_RESULTS_FILE, "w") as f:
        json.dump({
            "data": source, "stores": int(wide.shape[1]), "days": int(wide.shape[0]),
            "lookback_days": LOOKBACK, "horizon_days": HORIZON,
            "windows": {"train": int(len(train[0])), "validation": int(len(validation[0])), "test": int(len(test[0]))},
            "weights": int(model.count_params()), "rounds_trained": rounds,
            "test_scores": {"LSTM": lstm, **baselines},
            "verdict": verdict, "sample_store": int(test[2][0]), "sample_last_14_days": sample,
        }, f, indent=2)
    print("  Saved ->", DEMAND_LSTM_FILE, "and", DAY9_RESULTS_FILE)


if __name__ == "__main__":
    main()
