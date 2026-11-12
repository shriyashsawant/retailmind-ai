"""The RetailMind demand module: forecast how much each store will sell.

    from src.demand_service import forecast_next_day
    forecast_next_day(last_28_days_of_sales)        # one number: tomorrow's sales

Run from the main project folder (after src.train_day9):
    python -m src.demand_service                    # tomorrow's forecast for every store
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from src.config import DEMAND_LSTM_FILE
from src.demand_data import daily_table, load_sales, scale_windows
from src.train_day9 import LOOKBACK


def forecast_next_day(last_days):
    """last_days: the last 28 days of one store's sales (oldest first). Returns tomorrow's forecast in sales units."""
    last_days = np.asarray(last_days, dtype="float32")
    if len(last_days) != LOOKBACK:
        raise ValueError(f"The LSTM needs exactly {LOOKBACK} days of sales, but got {len(last_days)}.")
    scaled, _, average, spread = scale_windows(last_days[np.newaxis, :])           # same scaling as in training
    model = keras.models.load_model(DEMAND_LSTM_FILE)
    scaled_guess = model.predict(scaled[..., np.newaxis], verbose=0)[0, 0]
    return round(max(float(scaled_guess * spread[0, 0] + average[0, 0]), 0.0), 1)   # never below 0


if __name__ == "__main__":
    table, source = load_sales()
    wide = daily_table(table)
    print(f"Data: {source}")
    print(f"Forecast for {(wide.index[-1] + (wide.index[1] - wide.index[0])).date()}, from the last {LOOKBACK} days of each store")
    for store in wide.columns:
        print(f"  store {store}: {forecast_next_day(wide[store].to_numpy()[-LOOKBACK:])}")
