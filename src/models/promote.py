"""Train a candidate, compare it with the production model, promote if better.

The holdout is FIXED: the last --holdout-weeks weeks. No model trains on it.
The production model is re-scored on the same window, and the comparison is
refused if production has seen that window.

Run from the project root:
    python -m src.models.promote --tag cluster_style0.3
    python -m src.models.promote --tag cluster_style0.3 --max-depth 1
Exit code: 0 = decision made, 2 = comparison refused (leakage guard).
"""
import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src import tracking
from src.data.validate import validate_or_exit
from src.features.build import BASE_FEATURES, visual_features
from src.models.evaluate import mae, weighted_error
from src.models.rules import no_leak, should_promote
from src.models.xgb import PARAMS, QUANTILES, fit_models

PROCESSED = Path("data/processed")
NAME = "fashion-q{q}"


def holdout_split(df, holdout_weeks):
    weeks = np.sort(df["week"].unique())
    if not 1 <= holdout_weeks < len(weeks) - 52:
        raise ValueError("holdout must leave at least a year of training data")
    start = pd.Timestamp(weeks[-holdout_weeks])
    return (df["week"] < start).to_numpy(), (df["week"] >= start).to_numpy(), start


def score_models(models, df_te, features):
    """models: {q: fitted model}. Metrics on the holdout window."""
    X, y = df_te[features], df_te["sales"].to_numpy(dtype=float)
    P = np.column_stack([np.expm1(models[q].predict(X)).clip(min=0) for q in QUANTILES])
    p = dict(zip(QUANTILES, np.sort(P, axis=1).T))
    return {"WErr_q67": weighted_error(y, p[0.67]),
            "MAE_q50": mae(y, p[0.5]),
            "Cover80": float(((y >= p[0.1]) & (y <= p[0.9])).mean())}


def load_alias(client, alias):
    """Models, version, features and trained_until behind an alias, or None.
    The four quantile models are always registered together, so they share a version."""
    try:
        mv = client.get_model_version_by_alias(NAME.format(q=67), alias)
    except Exception:
        return None
    models = {q: mlflow.xgboost.load_model(f"models:/{NAME.format(q=int(q * 100))}@{alias}")
              for q in QUANTILES}
    features = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/features.json")["features"]
    return {"models": models, "version": int(mv.version), "features": features,
            "trained_until": mv.tags.get("trained_until")}


def register(client, models, trained_until, variant):
    """Log + register the 4 quantile models, tag each version, alias it 'staging'."""
    versions = {}
    for q, m in models.items():
        name = NAME.format(q=int(q * 100))
        info = mlflow.xgboost.log_model(m, name=f"model_q{int(q * 100)}")
        mv = mlflow.register_model(info.model_uri, name)
        client.set_model_version_tag(name, mv.version, "trained_until", trained_until)
        client.set_model_version_tag(name, mv.version, "variant", variant)
        client.set_registered_model_alias(name, "staging", mv.version)
        versions[q] = mv.version
    return versions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cluster_style0.3")
    ap.add_argument("--variant", default="base", choices=["base", "base+visual"])
    ap.add_argument("--holdout-weeks", type=int, default=12,
                    help="the last N weeks are scoring-only for every model")
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="candidate must lower WErr_q67 by at least this share")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in {"max_depth": args.max_depth,
                                   "n_estimators": args.n_estimators,
                                   "learning_rate": args.learning_rate}.items()
                 if v is not None}

    validate_or_exit("features", args.tag)
    feat_file = PROCESSED / f"features_{args.tag}.parquet"
    df = pd.read_parquet(feat_file).sort_values(["series_id", "week"]).reset_index(drop=True)
    features = BASE_FEATURES + (visual_features(df) if args.variant == "base+visual" else [])
    tr, te, holdout_start = holdout_split(df, args.holdout_weeks)
    trained_until = holdout_start.date().isoformat()
    df_tr, df_te = df[tr], df[te]
    print(f"train rows {len(df_tr):,} (weeks < {trained_until}) | "
          f"holdout rows {len(df_te):,}")

    mlflow.set_tracking_uri(tracking.TRACKING_URI)
    client = MlflowClient()
    prod = load_alias(client, "production")

    # guard: refuse to compare if production has seen the holdout
    if prod is not None:
        ok, why = no_leak(prod["trained_until"], holdout_start)
        if not ok:
            print(f"\nCOMPARISON REFUSED: {why}.")
            sys.exit(2)

    cand = fit_models(df_tr[features], df_tr["sales"].to_numpy(), overrides)
    cand_score = score_models(cand, df_te, features)

    prod_score = None
    if prod is not None:
        prod_score = score_models(prod["models"], df_te, prod["features"])  # same window
    promote, gain = should_promote(
        cand_score["WErr_q67"], prod_score["WErr_q67"] if prod_score else None,
        args.min_gain)
    decision = ("promote (first model)" if prod is None
                else "promote" if promote else "keep production")

    print(f"\ncandidate  ({args.variant}, {overrides or 'default params'}): {cand_score}")
    if prod_score:
        print(f"production (v{prod['version']}): {prod_score}")
        print(f"relative gain in WErr_q67: {gain:+.2%}  (needs >= {args.min_gain:.0%})")
    print(f"DECISION: {decision}")

    params = {"tag": args.tag, "variant": args.variant,
              "holdout_weeks": args.holdout_weeks, "trained_until": trained_until,
              "min_gain": args.min_gain, "n_features": len(features),
              **{f"xgb.{k}": v for k, v in {**PARAMS, **overrides}.items()}}
    with tracking.run("fashion-registry", f"candidate_{args.tag}_{args.variant}",
                      params=params, data_files=[feat_file],
                      tags={"decision": decision, "trained_until": trained_until}):
        mlflow.log_dict({"features": features}, "features.json")
        tracking.log_metrics({f"cand_{k}": v for k, v in cand_score.items()})
        if prod_score:
            tracking.log_metrics({f"prod_{k}": v for k, v in prod_score.items()})
            if gain is not None:
                tracking.log_metrics({"gain_WErr_q67": gain})
        versions = register(client, cand, trained_until, args.variant)
        if promote:
            for q, v in versions.items():
                client.set_registered_model_alias(NAME.format(q=int(q * 100)),
                                                  "production", v)
            print(f"promoted versions {versions} to 'production'")
        else:
            print(f"candidate stays in 'staging' (versions {versions})")


if __name__ == "__main__":
    main()