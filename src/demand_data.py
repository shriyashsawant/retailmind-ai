"""Days 9 and 10: get daily store sales, and cut them into (history -> future) windows for a sequence model.

A sales history is one long list of numbers, one per day. A sequence model learns from many short
"windows" cut out of it: the last LOOKBACK days are the question, the next HORIZON days are the answer.

Where the sales come from:
  - If the real Kaggle file "Rossmann Store Sales" (train.csv) is at data/raw/rossmann/train.csv, we use it.
  - If not, make_sales() creates MADE-UP sales for 8 stores with the patterns real shops show.
    Scores on made-up sales say whether the code works. They are NOT evidence about real demand.
"""
import os

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, SALES_FILE

STORE_COUNT = 8
DAYS = 960                    # about 2.6 years of daily sales
START_DATE = "2013-01-01"
REAL_STORE_LIMIT = 20         # the real file has 1,115 stores; the first 20 keep training quick

VAL_DAYS = 56                 # the last 8 weeks are the TEST period, the 8 weeks before that are VALIDATION
TEST_DAYS = 56


def make_sales():
    """Made-up daily sales for 8 stores. Returns a table with columns Store, Date, Sales."""
    random = np.random.RandomState(RANDOM_STATE)
    dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
    day = np.arange(DAYS)
    weekday_effect = np.array([1.15, 1.00, 0.95, 1.00, 1.10, 1.25, 0.0])    # Monday ... Sunday (shops closed on Sunday)
    rows = []
    for store in range(1, STORE_COUNT + 1):
        level = 3000 + 1500 * random.rand()                                  # each store has its own size
        yearly = 1 + 0.12 * np.sin(2 * np.pi * day / 365 + random.rand() * 6)
        trend = 1 + 0.0002 * day                                             # slow growth
        promo_start = random.randint(0, 14)
        on_promo = ((day + promo_start) % 14 < 5)                            # a promotion on 5 days of every 2 weeks
        promo_effect = np.where(on_promo, 1.20, 1.0)
        noise = np.exp(random.normal(0, 0.06, DAYS))
        sales = level * yearly * trend * promo_effect * noise * weekday_effect[dates.dayofweek]
        closed = (dates.month == 12) & (dates.day == 25) | (dates.month == 1) & (dates.day == 1)
        sales = np.where(closed, 0, sales)
        for date, value in zip(dates, sales):
            rows.append((store, date, round(float(value))))
    return pd.DataFrame(rows, columns=["Store", "Date", "Sales"])


def load_sales():
    """Returns (table with Store, Date, Sales, a short text saying where the data came from)."""
    if os.path.exists(SALES_FILE):
        table = pd.read_csv(SALES_FILE, usecols=["Store", "Date", "Sales"], parse_dates=["Date"])
        table = table[table["Store"] <= REAL_STORE_LIMIT]
        return table, "real Rossmann Store Sales (first %d stores)" % REAL_STORE_LIMIT
    return make_sales(), "MADE-UP sales for %d stores (the real Rossmann file was not found)" % STORE_COUNT


def daily_table(table):
    """One column per store, one row per day, with missing days filled in as 0 sales. Returns a DataFrame."""
    wide = table.pivot_table(index="Date", columns="Store", values="Sales", aggfunc="sum")
    wide = wide.asfreq("D").fillna(0)       # days with no row (shop closed or missing) count as 0 sales
    return wide


def make_windows(wide, lookback, horizon, first_target, last_target_end):
    """Cut every store's sales into windows.

    X = the `lookback` days before day t, y = the `horizon` days starting at day t.
    Only windows whose answer (day t up to t + horizon) lies between `first_target` and `last_target_end`
    are kept, so training windows never contain answers from the validation or test weeks.
    Returns X (windows, lookback), y (windows, horizon) in real sales units, and the store of each window.
    """
    values = wide.to_numpy(dtype="float32")
    stores = list(wide.columns)
    X, y, window_store = [], [], []
    for column, store in enumerate(stores):
        series = values[:, column]
        for t in range(max(first_target, lookback), last_target_end - horizon + 1):
            X.append(series[t - lookback:t])
            y.append(series[t:t + horizon])
            window_store.append(store)
    return np.array(X, dtype="float32"), np.array(y, dtype="float32"), np.array(window_store)


def split_windows(wide, lookback, horizon):
    """Chronological split: train (oldest), validation (8 weeks), test (the most recent 8 weeks)."""
    n = len(wide)
    train = make_windows(wide, lookback, horizon, 0, n - VAL_DAYS - TEST_DAYS)
    validation = make_windows(wide, lookback, horizon, n - VAL_DAYS - TEST_DAYS, n - TEST_DAYS)
    test = make_windows(wide, lookback, horizon, n - TEST_DAYS, n)
    return train, validation, test


def scale_windows(X, y=None):
    """Scale each window by ITS OWN average and spread, so one model can serve small and big stores.

    Returns the scaled X, the scaled y (if given), and the average and spread of each window
    (needed to turn a forecast back into real sales).
    """
    average = X.mean(axis=1, keepdims=True)
    spread = X.std(axis=1, keepdims=True) + 1e-6
    scaled_x = (X - average) / spread
    scaled_y = None if y is None else (y - average) / spread
    return scaled_x, scaled_y, average, spread


def mean_absolute_error(real, guess):
    """On average, how many sales units the forecast is off."""
    return float(np.mean(np.abs(real - guess)))


def root_mean_squared_error(real, guess):
    """Like the average error, but big misses count extra."""
    return float(np.sqrt(np.mean((real - guess) ** 2)))
