"""Metrics and rolling-origin backtest shared by every model."""
import numpy as np
import pandas as pd


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def weighted_error(y, yhat, c_under=2.0, c_over=1.0):
    """Business cost: under-forecasting (stockout, lost sale) costs
    c_under per unit, over-forecasting (overstock, markdown) costs c_over."""
    under = np.maximum(y - yhat, 0)
    over = np.maximum(yhat - y, 0)
    return float(np.mean(c_under * under + c_over * over))


def rolling_origin(n_weeks, initial=104, horizon=4, step=4):
    """Yield (train_end, test_end). Train on weeks [0, train_end),
    test on weeks [train_end, test_end). The training window grows."""
    train_end = initial
    while train_end + horizon <= n_weeks:
        yield train_end, train_end + horizon
        train_end += step


def run_backtest(weekly, forecast_fn, initial=104, horizon=4, step=4,
                 key="articleType"):
    """weekly: DataFrame with <key>, week, sales.
    forecast_fn(train: np.ndarray, horizon: int) -> np.ndarray
    The function only receives the training part, so it cannot peek."""
    rows = []
    for s, g in weekly.groupby(key):
        g = g.sort_values("week")
        y = g["sales"].to_numpy(dtype=float)
        weeks = g["week"].to_numpy()
        for fold, (a, b) in enumerate(rolling_origin(len(y), initial, horizon, step)):
            yhat = forecast_fn(y[:a], horizon)
            for k in range(horizon):
                rows.append((s, fold, weeks[a + k], y[a + k], yhat[k]))
    return pd.DataFrame(rows, columns=["series", "fold", "week", "y", "yhat"])


def summarize(res):
    """Overall and per-fold metrics (std across folds = error bar)."""
    def metrics(d):
        return pd.Series({"MAE": mae(d.y, d.yhat),
                          "RMSE": rmse(d.y, d.yhat),
                          "WErr": weighted_error(d.y, d.yhat)})
    overall = metrics(res)
    per_fold = res.groupby("fold").apply(metrics)
    return overall, per_fold.mean(), per_fold.std()