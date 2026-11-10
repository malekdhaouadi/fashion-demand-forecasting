"""Seasonal-naive baseline.  Run:  python -m src.models.baseline"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluate import run_backtest, summarize

PROCESSED = Path("data/processed")


def seasonal_naive(train: np.ndarray, horizon: int, season: int = 52) -> np.ndarray:
    n = len(train)
    idx = n + np.arange(horizon) - season      # same week last year
    return train[idx]


def main(tag: str = "style0.3") -> None:
    weekly = pd.read_parquet(PROCESSED / f"category_weekly_{tag}.parquet")
    res = run_backtest(weekly, seasonal_naive)
    overall, mean_f, std_f = summarize(res)

    print(f"Dataset: {tag}  |  folds: {res.fold.nunique()}")
    print("\nOverall:\n", overall.round(1))
    print("\nPer-fold mean +/- std:")
    for m in overall.index:
        print(f"  {m}: {mean_f[m]:.1f} +/- {std_f[m]:.1f}")

    per_cat = res.groupby("articleType").apply(
        lambda d: np.mean(np.abs(d.y - d.yhat)))
    print("\nMAE per category:\n", per_cat.round(1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "style0.3")