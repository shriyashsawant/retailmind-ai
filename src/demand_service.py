"""The RetailMind demand module: forecast how much each store will sell.

    from src.demand_service import forecast_next_day
    forecast_next_day(last_28_days_of_sales)        # Day 9: one number, tomorrow's sales
    forecast_weeks(last_56_days_of_sales)           # Day 10: the next 4 weeks, daily and weekly totals

Run from the main project folder (after src.train_day9 and src.train_day10):
    python -m src.demand_service                    # forecasts for every store
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from src.config import DEMAND_LSTM_FILE, SEQ2SEQ_FILE
from src.demand_data import daily_table, load_sales, scale_windows
from src.train_day9 import LOOKBACK
from src.train_day10 import HORIZON as SEQ2SEQ_DAYS, LOOKBACK as SEQ2SEQ_LOOKBACK


def forecast_next_day(last_days):
    """last_days: the last 28 days of one store's sales (oldest first). Returns tomorrow's forecast in sales units."""
    last_days = np.asarray(last_days, dtype="float32")
    if len(last_days) != LOOKBACK:
        raise ValueError(f"The LSTM needs exactly {LOOKBACK} days of sales, but got {len(last_days)}.")
    scaled, _, average, spread = scale_windows(last_days[np.newaxis, :])           # same scaling as in training
    model = keras.models.load_model(DEMAND_LSTM_FILE)
    scaled_guess = model.predict(scaled[..., np.newaxis], verbose=0)[0, 0]
    return round(max(float(scaled_guess * spread[0, 0] + average[0, 0]), 0.0), 1)   # never below 0


def forecast_weeks(last_days, weeks=4):
    """Day 10: forecast the next 1 to 4 weeks with the Seq2Seq + attention model.

    last_days: the last 56 days of one store's sales (oldest first).
    Returns the forecast for every day and the total for each week (handy for ordering stock).
    """
    last_days = np.asarray(last_days, dtype="float32")
    if len(last_days) != SEQ2SEQ_LOOKBACK:
        raise ValueError(f"The Seq2Seq model needs exactly {SEQ2SEQ_LOOKBACK} days of sales, but got {len(last_days)}.")
    if not 1 <= weeks <= SEQ2SEQ_DAYS // 7:
        raise ValueError(f"weeks must be between 1 and {SEQ2SEQ_DAYS // 7}.")
    scaled, _, average, spread = scale_windows(last_days[np.newaxis, :])
    model = keras.models.load_model(SEQ2SEQ_FILE)
    daily = np.maximum(model.predict(scaled[..., np.newaxis], verbose=0)[0] * spread[0, 0] + average[0, 0], 0)
    daily = daily[:weeks * 7]
    return {"daily": [round(float(v), 1) for v in daily],
            "weekly_totals": [round(float(daily[w * 7:(w + 1) * 7].sum()), 1) for w in range(weeks)]}


if __name__ == "__main__":
    table, source = load_sales()
    wide = daily_table(table)
    print(f"Data: {source}")
    print(f"Forecast for {(wide.index[-1] + (wide.index[1] - wide.index[0])).date()}, from the last {LOOKBACK} days of each store")
    for store in wide.columns:
        print(f"  store {store}: {forecast_next_day(wide[store].to_numpy()[-LOOKBACK:])}")
    print(f"\nNext 4 weeks (weekly totals), from the last {SEQ2SEQ_LOOKBACK} days of each store")
    for store in wide.columns:
        print(f"  store {store}: {forecast_weeks(wide[store].to_numpy()[-SEQ2SEQ_LOOKBACK:])['weekly_totals']}")
