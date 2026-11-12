"""Day 10: forecast the next 4 weeks at once with a Seq2Seq model that uses attention.

Day 9 predicted ONE day. To plan stock for a month we need many days ahead. Two ways to get there:
  - roll the Day 9 model forward: predict tomorrow, feed that guess back in, predict the day after ...
    (small mistakes get fed back in and can pile up)
  - Seq2Seq: an ENCODER reads 8 weeks of history, a DECODER writes the next 28 days in one go, and
    ATTENTION lets each forecast day look back at the past days that matter most for it.

Steps:
    1. Windows: the last 56 days -> the next 28 days. Same time-based split as Day 9.
    2. Baselines and the Day 9 LSTM rolled forward 28 times.
    3. Train Seq2Seq with attention, and the same network WITHOUT attention (to see if attention helps).
    4. Compare the error for week 1, 2, 3 and 4 of the forecast.
    5. Look at the attention weights, then save the model and a picture.

Run from the main project folder (after src.train_day9):   python -m src.train_day10
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow's chatty startup messages

import cv2
import keras
import numpy as np

from src.config import (ATTENTION_IMAGE, DAY10_RESULTS_FILE, DEMAND_LSTM_FILE, MODEL_FOLDER,
                        RANDOM_STATE, SEQ2SEQ_FILE)
from src.demand_data import (daily_table, load_sales, mean_absolute_error, root_mean_squared_error,
                             scale_windows, split_windows)
from src.train_day9 import LOOKBACK as DAY9_LOOKBACK

LOOKBACK = 56      # the encoder reads the last 8 weeks
HORIZON = 28       # the decoder writes the next 4 weeks


def build_seq2seq(use_attention=True):
    """Encoder-decoder for forecasting. Shapes: 56 days in -> 28 days out.

    - Encoder: an LSTM that reads the 56 days. It keeps its notes for EVERY day (encoded_days)
      and a short summary of the whole history (the final state).
    - Decoder: an LSTM that starts from that summary and writes 28 forecast steps.
    - Attention: each of the 28 forecast steps looks across all 56 past days and picks how much to listen
      to each one (the weights add up to 1). Without it, the decoder only has the single summary.
    This decoder is not fed its own earlier guesses, so there is no teacher forcing and no step-by-step loop.
    """
    history = keras.Input(shape=(LOOKBACK, 1))
    encoded_days, state_h, state_c = keras.layers.LSTM(32, return_sequences=True, return_state=True)(history)

    decoder_input = keras.layers.RepeatVector(HORIZON)(state_h)             # one copy of the summary per forecast day
    decoded = keras.layers.LSTM(32, return_sequences=True)(decoder_input, initial_state=[state_h, state_c])

    attention_weights = None
    if use_attention:
        looked_up, attention_weights = keras.layers.Attention()([decoded, encoded_days], return_attention_scores=True)
        decoded = keras.layers.Concatenate()([decoded, looked_up])          # decoder notes + what it looked up

    forecast = keras.layers.Dense(1)(decoded)                              # one number per forecast day
    forecast = keras.layers.Reshape((HORIZON,))(forecast)

    model = keras.Model(history, forecast)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    attention_model = keras.Model(history, attention_weights) if use_attention else None
    return model, attention_model


def train(model, x_train, y_train, x_val, y_val):
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(x_train[..., np.newaxis], y_train, validation_data=(x_val[..., np.newaxis], y_val),
                        epochs=150, batch_size=64, callbacks=[early_stop], verbose=0)   # a high cap: early stopping decides
    return len(history.history["val_loss"]), min(history.history["val_loss"])


def day9_rolled_forward(day9_model, history, steps):
    """Use the Day 9 one-day model to forecast `steps` days: predict a day, add it to the history, repeat."""
    recent = history[:, -DAY9_LOOKBACK:].copy()
    forecasts = []
    for _ in range(steps):
        scaled, _, average, spread = scale_windows(recent)
        guess = day9_model.predict(scaled[..., np.newaxis], verbose=0)[:, 0] * spread[:, 0] + average[:, 0]
        guess = np.maximum(guess, 0)
        forecasts.append(guess)
        recent = np.concatenate([recent[:, 1:], guess[:, np.newaxis]], axis=1)       # slide the window by one day
    return np.stack(forecasts, axis=1)


def error_by_week(real, guess):
    """MAE for forecast week 1, 2, 3 and 4, plus the overall MAE and RMSE."""
    weeks = [round(mean_absolute_error(real[:, w * 7:(w + 1) * 7], guess[:, w * 7:(w + 1) * 7]), 1) for w in range(4)]
    return {"mae": round(mean_absolute_error(real, guess), 1), "rmse": round(root_mean_squared_error(real, guess), 1),
            "mae_by_week": weeks}


def attention_picture(average_attention, path):
    """Save a heatmap: rows = forecast day 1..28 (top to bottom), columns = past day 56 days ago .. yesterday."""
    scaled = average_attention / average_attention.max(axis=1, keepdims=True)           # each row 0..1 so every row is visible
    heat = cv2.applyColorMap(cv2.resize((scaled * 255).astype("uint8"), None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST),
                             cv2.COLORMAP_VIRIDIS)
    canvas = np.full((heat.shape[0] + 50, heat.shape[1] + 70, 3), 255, np.uint8)
    canvas[10:10 + heat.shape[0], 60:60 + heat.shape[1]] = heat
    for lag in range(7, LOOKBACK + 1, 7):                                                # a tick at 7, 14, 21 ... days back
        x = 60 + (LOOKBACK - lag) * 12 + 6
        cv2.line(canvas, (x, 10 + heat.shape[0]), (x, 16 + heat.shape[0]), (0, 0, 0), 1)
        cv2.putText(canvas, str(lag), (x - 8, 31 + heat.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "days before the forecast starts  (older <-  -> newer)", (60, 45 + heat.shape[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    for day in (1, 7, 14, 21, 28):
        cv2.putText(canvas, f"day {day}", (4, 10 + (day - 1) * 12 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas)


def main():
    print("Step 1: windows (56 days -> next 28 days), split by time")
    table, source = load_sales()
    wide = daily_table(table)
    print(f"  {source}")
    train_set, val_set, test_set = split_windows(wide, LOOKBACK, HORIZON)
    print(f"  train {len(train_set[0])} | validation {len(val_set[0])} | test {len(test_set[0])} windows")
    x_train, y_train, _, _ = scale_windows(*train_set[:2])
    x_val, y_val, _, _ = scale_windows(*val_set[:2])
    x_test, _, test_average, test_spread = scale_windows(test_set[0])
    history_raw, real = test_set[0], test_set[1]

    print("\nStep 2: baselines and the Day 9 LSTM rolled forward 28 times")
    day = np.arange(HORIZON)
    scores = {
        "same weekday last week": error_by_week(real, history_raw[:, -7 + (day % 7)]),
        "same weekday 2 weeks ago": error_by_week(real, history_raw[:, -14 + (day % 14)]),
        "Day 9 LSTM rolled forward": error_by_week(real, day9_rolled_forward(keras.models.load_model(DEMAND_LSTM_FILE), history_raw, HORIZON)),
    }

    print("\nStep 3: train Seq2Seq with attention, and the same network without attention")
    models, rounds = {}, {}
    attention_model = None
    for name, use_attention in [("Seq2Seq without attention", False), ("Seq2Seq with attention", True)]:
        keras.utils.set_random_seed(RANDOM_STATE)              # the same starting point every run
        model, attention = build_seq2seq(use_attention)
        rounds[name], best_loss = train(model, x_train, y_train, x_val, y_val)
        print(f"  {name}: {model.count_params()} weights, {rounds[name]} rounds, best validation loss {best_loss:.4f}")
        guess = model.predict(x_test[..., np.newaxis], verbose=0) * test_spread + test_average
        scores[name] = error_by_week(real, np.maximum(guess, 0))
        models[name] = model
        if use_attention:
            attention_model = attention

    print("\nStep 4: forecast error for the next 28 days (test windows), in sales units")
    print(f"  {'Model':<28}{'MAE':>8}{'RMSE':>8}   MAE week 1 / 2 / 3 / 4")
    for name, s in scores.items():
        print(f"  {name:<28}{s['mae']:>8}{s['rmse']:>8}   " + " / ".join(str(w) for w in s["mae_by_week"]))
    best = min(scores, key=lambda name: scores[name]["mae"])
    print("  -> lowest MAE:", best)

    print("\nStep 5: attention weights (which past days each forecast day looked at)")
    weights = attention_model.predict(x_test[..., np.newaxis], verbose=0)       # (windows, 28 forecast days, 56 past days)
    average_attention = weights.mean(axis=0)
    overall = average_attention.mean(axis=0)                                    # how much each past day was used, on average
    top = overall.argsort()[::-1][:5]
    top_lags = [int(LOOKBACK - position) for position in top]                   # position 55 is 1 day back, position 0 is 56 days back
    weekly_lags = [lag for lag in range(7, LOOKBACK + 1, 7)]
    weekly_share = float(sum(overall[LOOKBACK - lag] for lag in weekly_lags))
    even_share = len(weekly_lags) / LOOKBACK
    print("  the 5 past days used most (days before the forecast starts):", top_lags)
    print(f"  share of attention on 7, 14, 21 ... days back: {weekly_share:.3f} (an even spread would give {even_share:.3f})")
    attention_picture(average_attention, ATTENTION_IMAGE)

    os.makedirs(MODEL_FOLDER, exist_ok=True)
    models["Seq2Seq with attention"].save(SEQ2SEQ_FILE)
    with open(DAY10_RESULTS_FILE, "w") as f:
        json.dump({
            "data": source, "lookback_days": LOOKBACK, "horizon_days": HORIZON,
            "windows": {"train": int(len(train_set[0])), "validation": int(len(val_set[0])), "test": int(len(test_set[0]))},
            "rounds_trained": rounds, "test_scores": scores, "lowest_mae": best,
            "attention": {"most_used_days_back": top_lags,
                          "share_on_weekly_lags": round(weekly_share, 3), "share_if_even": round(even_share, 3)},
        }, f, indent=2)
    print("\n  Saved ->", SEQ2SEQ_FILE, ",", DAY10_RESULTS_FILE, "and", ATTENTION_IMAGE)


if __name__ == "__main__":
    main()
