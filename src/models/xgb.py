"""Global XGBoost quantile model with a rolling-origin ablation.

Run from the project root:
    python -m src.models.xgb --tag cluster_style0.3
    python -m src.models.xgb --tag cluster_style0.0
Output: data/processed/preds_<tag>.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features.build import BASE_FEATURES, visual_features
from src.models.evaluate import mae, rmse, rolling_origin, weighted_error

PROCESSED = Path("data/processed")

# 0.67 = cost-optimal quantile when a stockout costs 2x an overstock (2 / (2 + 1))
QUANTILES = [0.1, 0.5, 0.67, 0.9]
PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              random_state=42, n_jobs=-1)


def fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    """One model per quantile. Trained on log1p(sales): quantiles are preserved
    by a monotone transform, and it puts small and large series on one scale."""
    cols = []
    for q in QUANTILES:
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **PARAMS)
        m.fit(Xtr, np.log1p(ytr))
        cols.append(np.expm1(m.predict(Xte)).clip(min=0))
    return np.sort(np.column_stack(cols), axis=1)      # no crossing quantiles


def backtest(df, features, name, initial, horizon, step) -> pd.DataFrame:
    weeks = np.sort(df["week"].unique())
    t = df["week"].map({w: i for i, w in enumerate(weeks)}).to_numpy()
    out = []
    for fold, (a, b) in enumerate(rolling_origin(len(weeks), initial, horizon, step)):
        tr, te = t < a, (t >= a) & (t < b)            # train strictly before the origin
        P = fit_predict(df.loc[tr, features], df.loc[tr, "sales"].to_numpy(),
                        df.loc[te, features])
        d = df.loc[te, ["series_id", "week", "sales"]].rename(columns={"sales": "y"}).copy()
        for j, q in enumerate(QUANTILES):
            d[f"q{int(q * 100)}"] = P[:, j]
        d["fold"], d["variant"] = fold, name
        out.append(d)
    return pd.concat(out, ignore_index=True)


def naive(df, initial, horizon, step) -> pd.DataFrame:
    """Seasonal naive = the lag_52 column. Sanity check against baseline.py."""
    weeks = np.sort(df["week"].unique())
    t = df["week"].map({w: i for i, w in enumerate(weeks)}).to_numpy()
    out = []
    for fold, (a, b) in enumerate(rolling_origin(len(weeks), initial, horizon, step)):
        te = (t >= a) & (t < b)
        d = df.loc[te, ["series_id", "week", "sales"]].rename(columns={"sales": "y"}).copy()
        d["q50"] = df.loc[te, "lag_52"].to_numpy()
        d["fold"], d["variant"] = fold, "seasonal_naive"
        out.append(d)
    return pd.concat(out, ignore_index=True)


def fold_metrics(d: pd.DataFrame) -> pd.Series:
    y, p50 = d["y"].to_numpy(), d["q50"].to_numpy()
    m = {"MAE": mae(y, p50), "RMSE": rmse(y, p50), "WErr_q50": weighted_error(y, p50)}
    if "q67" in d:
        m["WErr_q67"] = weighted_error(y, d["q67"].to_numpy())
        m["Cover80"] = float(((y >= d["q10"]) & (y <= d["q90"])).mean())
    return pd.Series(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cluster_style0.3")
    ap.add_argument("--initial", type=int, default=104)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--step", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / f"features_{args.tag}.parquet")
    df = df.sort_values(["series_id", "week"]).reset_index(drop=True)
    df["series_code"] = df["series_id"].astype("category").cat.codes

    variants = {
        "base": BASE_FEATURES,
        # control: the visual features are constant per series, so they can act
        # as a series identifier. This variant gets the identity but no visual info.
        "base+series_id": BASE_FEATURES + ["series_code"],
        "base+visual": BASE_FEATURES + visual_features(df),
    }

    parts = [naive(df, args.initial, args.horizon, args.step)]
    for name, feats in variants.items():
        print(f"running {name} ({len(feats)} features) ...")
        parts.append(backtest(df, feats, name, args.initial, args.horizon, args.step))
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(PROCESSED / f"preds_{args.tag}.parquet", index=False)

    # per-fold metrics -> mean +/- std across folds
    pf = pd.DataFrame([fold_metrics(g).rename((v, f))
                       for (v, f), g in res.groupby(["variant", "fold"])])
    pf.index = pd.MultiIndex.from_tuples(pf.index, names=["variant", "fold"])
    mean, std = pf.groupby("variant").mean(), pf.groupby("variant").std()
    order = ["seasonal_naive"] + list(variants)
    print(f"\n=== {args.tag} | {res.fold.nunique()} folds ===")
    table = mean.loc[order].round(1).astype(str) + " +/- " + std.loc[order].round(1).astype(str)
    print(table.to_string())

    # paired comparison on the same folds (negative diff = better than 'base')
    print("\nPaired MAE difference vs base (per fold):")
    base = pf.loc["base", "MAE"].to_numpy()
    for v in ["base+series_id", "base+visual"]:
        diff = pf.loc[v, "MAE"].to_numpy() - base
        print(f"  {v:15s} mean {diff.mean():+.2f}  std {diff.std(ddof=1):.2f}  "
              f"better in {(diff < 0).sum()}/{len(diff)} folds")
    diff = pf.loc["base+visual", "MAE"].to_numpy() - pf.loc["base+series_id", "MAE"].to_numpy()
    print(f"  visual vs series_id control: mean {diff.mean():+.2f}  "
          f"better in {(diff < 0).sum()}/{len(diff)} folds")


if __name__ == "__main__":
    main()