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
import mlflow
import mlflow.xgboost

from src import tracking

from src.features.build import BASE_FEATURES, visual_features
from src.models.evaluate import mae, rmse, rolling_origin, weighted_error
from src.data.validate import validate_or_exit

PROCESSED = Path("data/processed")

# 0.67 = cost-optimal quantile when a stockout costs 2x an overstock (2 / (2 + 1))
QUANTILES = [0.1, 0.5, 0.67, 0.9]
PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
              random_state=42, n_jobs=-1)


def fit_models(Xtr, ytr) -> dict:
    models = {}
    for q in QUANTILES:
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **PARAMS)
        m.fit(Xtr, np.log1p(ytr))
        models[q] = m
    return models


def fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    """One model per quantile. Trained on log1p(sales): quantiles are preserved
    by a monotone transform, and it puts small and large series on one scale."""
    models = fit_models(Xtr, ytr)
    cols = [np.expm1(models[q].predict(Xte)).clip(min=0) for q in QUANTILES]
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
    if d["q67"].notna().all():                      # only the quantile models have it
        m["WErr_q67"] = weighted_error(y, d["q67"].to_numpy())
        m["Cover80"] = float(((y >= d["q10"]) & (y <= d["q90"])).mean())
    return pd.Series(m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cluster_style0.3")
    ap.add_argument("--initial", type=int, default=104)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--log-model", action="store_true",
                    help="also fit on ALL weeks and log the 4 quantile models")
    args = ap.parse_args()
    validate_or_exit("features", args.tag)

    feat_file = PROCESSED / f"features_{args.tag}.parquet"
    df = pd.read_parquet(feat_file)
    df = df.sort_values(["series_id", "week"]).reset_index(drop=True)
    df["series_code"] = df["series_id"].astype("category").cat.codes

    variants = {
        "base": BASE_FEATURES,
        "base+series_id": BASE_FEATURES + ["series_code"],
        "base+visual": BASE_FEATURES + visual_features(df),
    }

    parts = [naive(df, args.initial, args.horizon, args.step)]
    for name, feats in variants.items():
        print(f"running {name} ({len(feats)} features) ...")
        parts.append(backtest(df, feats, name, args.initial, args.horizon, args.step))
    res = pd.concat(parts, ignore_index=True)
    preds_file = PROCESSED / f"preds_{args.tag}.parquet"
    res.to_parquet(preds_file, index=False)

    # per-fold metrics -> mean +/- std across folds
    pf = pd.DataFrame([fold_metrics(g).rename((v, f))
                       for (v, f), g in res.groupby(["variant", "fold"])])
    pf.index = pd.MultiIndex.from_tuples(pf.index, names=["variant", "fold"])
    mean, std = pf.groupby("variant").mean(), pf.groupby("variant").std()
    order = ["seasonal_naive"] + list(variants)

    def fmt(v, c):
        if pd.isna(mean.loc[v, c]):
            return "-"
        d = 3 if c == "Cover80" else 1
        return f"{mean.loc[v, c]:.{d}f} +/- {std.loc[v, c]:.{d}f}"
    table = pd.DataFrame({c: [fmt(v, c) for v in order] for c in mean.columns}, index=order)
    print(f"\n=== {args.tag} | {res.fold.nunique()} folds ===")
    print(table.to_string())

    # paired comparison on the same folds (negative diff = better)
    print("\nPaired MAE difference (per fold):")
    base = pf.loc["base", "MAE"].to_numpy()
    paired = {}
    for v in ["base+series_id", "base+visual"]:
        diff = pf.loc[v, "MAE"].to_numpy() - base
        paired[f"paired_mae_{v.replace('+', '_')}_vs_base"] = diff.mean()
        print(f"  {v:15s} vs base: mean {diff.mean():+.2f}  std {diff.std(ddof=1):.2f}  "
              f"better in {(diff < 0).sum()}/{len(diff)} folds")
    diff = pf.loc["base+visual", "MAE"].to_numpy() - pf.loc["base+series_id", "MAE"].to_numpy()
    paired["paired_mae_visual_vs_series_id"] = diff.mean()
    print(f"  visual vs series_id control: mean {diff.mean():+.2f}  "
          f"better in {(diff < 0).sum()}/{len(diff)} folds")

    # ---------- MLflow ----------
    params = {"tag": args.tag, "initial": args.initial, "horizon": args.horizon,
              "step": args.step, "quantiles": QUANTILES, "n_folds": res.fold.nunique(),
              **{f"xgb.{k}": v for k, v in PARAMS.items()}}
    with tracking.run("fashion-xgb", f"xgb_{args.tag}", params=params,
                      data_files=[feat_file]):
        tracking.log_metrics(paired)
        mlflow.log_artifact(str(preds_file))
        for v in order:
            with tracking.run("fashion-xgb", v, nested=True, tags={"variant": v}):
                if v in variants:
                    mlflow.log_param("n_features", len(variants[v]))
                    mlflow.log_dict({"features": variants[v]}, "features.json")
                tracking.log_metrics({f"{c}_mean": mean.loc[v, c] for c in mean.columns})
                tracking.log_metrics({f"{c}_std": std.loc[v, c] for c in std.columns})
                for fold, row in pf.loc[v].iterrows():          # curve over folds
                    tracking.log_metrics({f"fold_{c}": row[c] for c in pf.columns}, step=int(fold))
                if args.log_model and v in variants:
                    models = fit_models(df[variants[v]], df["sales"].to_numpy())
                    for q, m in models.items():
                        mlflow.xgboost.log_model(m, artifact_path=f"model_q{int(q * 100)}")
    print("\nlogged to MLflow (run: mlflow ui --backend-store-uri sqlite:///mlflow.db)")


if __name__ == "__main__":
    main()