"""Data validation with pandera. A failed check stops the pipeline.

Run from the project root:
    python -m src.data.validate --stage all --tag cluster_style0.3
Exit code 0 = everything passed, 1 = at least one check failed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:                      # older pandera versions
    import pandera as pa

from src.data.load import KEEP
from src.data.simulate import N_WEEKS
from src.features.build import BASE_FEATURES

PROCESSED = Path("data/processed")
SEASONS = ["Summer", "Fall", "Winter", "Spring"]
GENDERS = ["Men", "Women", "Boys", "Girls", "Unisex"]


# ---------------------------------------------------------------- schemas
def products_schema(check_images: bool = True, min_per_type: int = 100):
    cols = {
        "id": pa.Column(int, pa.Check.ge(0), unique=True, coerce=True),
        "gender": pa.Column(str, pa.Check.isin(GENDERS)),
        "masterCategory": pa.Column(str),
        "articleType": pa.Column(str, pa.Check.isin(KEEP)),
        "baseColour": pa.Column(str),                       # not nullable
        "season": pa.Column(str, pa.Check.isin(SEASONS)),
        "usage": pa.Column(str),
        "year": pa.Column(int, pa.Check.in_range(2007, 2019), coerce=True),
        "image_path": pa.Column(str),
    }
    if check_images:
        cols["image_path"] = pa.Column(str, pa.Check(
            lambda s: s.map(lambda p: Path(p).exists()),
            error="image file missing on disk"))
    return pa.DataFrameSchema(cols, checks=[
        pa.Check(lambda df: set(df["articleType"]) == set(KEEP),
                 error="some articleType in KEEP has no products"),
        pa.Check(lambda df: df.groupby("articleType").size().min() >= min_per_type,
                 error=f"fewer than {min_per_type} products in some category"),
    ])


def series_schema(n_weeks: int = N_WEEKS):
    return pa.DataFrameSchema({
        "series_id": pa.Column(str),
        "articleType": pa.Column(str, pa.Check.isin(KEEP)),
        "style_group": pa.Column(),
        "week": pa.Column("datetime64[ns]", coerce=True),
        "sales": pa.Column(int, [pa.Check.ge(0), pa.Check.le(100_000)], coerce=True),
        "price": pa.Column(float, pa.Check.in_range(1, 500)),
        "promo": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
    }, checks=[
        pa.Check(lambda df: not df.duplicated(["series_id", "week"]).any(),
                 error="duplicate (series_id, week)"),
        pa.Check(lambda df: bool((df.groupby("series_id").size() == n_weeks).all()),
                 error=f"every series must have exactly {n_weeks} weeks"),
        pa.Check(lambda df: all(
            (g.sort_values("week")["week"].diff().dropna() == pd.Timedelta(weeks=1)).all()
            for _, g in df.groupby("series_id")),
            error="weeks are not consecutive"),
        pa.Check(lambda df: bool((df.groupby("series_id")["sales"].sum() > 0).all()),
                 error="a series has zero sales over the whole period"),
    ])


def features_schema():
    no_inf = pa.Check(lambda s: ~np.isinf(s), error="inf value")
    cols = {c: pa.Column(float, no_inf, nullable=True, coerce=True)
            for c in BASE_FEATURES}
    cols["^vis_"] = pa.Column(float, regex=True, nullable=False, coerce=True)
    cols["sales"] = pa.Column(float, pa.Check.ge(0), coerce=True)
    season_cols = [f"season_{s}" for s in SEASONS]
    return pa.DataFrameSchema(cols, checks=[
        pa.Check(lambda df: not df.duplicated(["series_id", "week"]).any(),
                 error="duplicate (series_id, week)"),
        pa.Check(lambda df: df["lag_4"].isna().mean() <= 0.10,
                 error="too many missing lag_4 (expected about 3%)"),
        pa.Check(lambda df: df["lag_52"].isna().mean() <= 0.40,
                 error="too many missing lag_52 (expected about 33%)"),
        pa.Check(lambda df: np.allclose(df[season_cols].sum(axis=1), 1, atol=1e-6),
                 error="season shares do not sum to 1"),
    ])


# ---------------------------------------------------------------- validators
def validate_products(df, check_images=True, min_per_type=100):
    return products_schema(check_images, min_per_type).validate(df, lazy=True)


def validate_series(df, n_weeks=N_WEEKS):
    return series_schema(n_weeks).validate(df, lazy=True)


def validate_features(df):
    return features_schema().validate(df, lazy=True)


def validate_embeddings(ids_emb, X, ids_products, dim=512):
    if X.shape != (len(ids_products), dim):
        raise ValueError(f"embeddings shape {X.shape}, expected {(len(ids_products), dim)}")
    if not np.isfinite(X).all():
        raise ValueError("NaN or inf in embeddings")
    if not (np.asarray(ids_emb) == np.asarray(ids_products)).all():
        raise ValueError("embedding ids do not line up with products.csv")
    if (np.linalg.norm(X, axis=1) == 0).any():
        raise ValueError("an embedding is all zeros")


def validate_clusters(clusters, products):
    if set(clusters["id"]) != set(products["id"]):
        raise ValueError("clusters.csv and products.csv do not have the same ids")
    if clusters["cluster"].isna().any():
        raise ValueError("missing cluster id")
    if clusters.groupby("articleType")["cluster"].nunique().min() < 2:
        raise ValueError("a category has fewer than 2 clusters")


# ---------------------------------------------------------------- runner
def _run(name, fn, failures):
    try:
        fn()
        print(f"  OK    {name}")
    except pa.errors.SchemaErrors as e:
        failures.append(name)
        print(f"  FAIL  {name}")
        cols = [c for c in ["column", "check", "failure_case"] if c in e.failure_cases]
        print(e.failure_cases[cols].head(10).to_string(index=False))
    except (ValueError, AssertionError, FileNotFoundError, KeyError) as e:
        failures.append(name)
        print(f"  FAIL  {name}: {e}")


def validate_stage(stage: str, tag: str) -> list:
    failures = []
    stages = ["raw", "embeddings", "sales", "features"] if stage == "all" else [stage]
    print(f"validating stage(s): {', '.join(stages)}  (tag={tag})")

    if "raw" in stages:
        _run("products.csv", lambda: validate_products(
            pd.read_csv(PROCESSED / "products.csv")), failures)

    if "embeddings" in stages:
        def emb():
            products = pd.read_csv(PROCESSED / "products.csv")
            z = np.load(PROCESSED / "embeddings.npz")
            validate_embeddings(z["ids"], z["X"], products["id"].to_numpy())
        _run("embeddings.npz", emb, failures)
        _run("clusters.csv", lambda: validate_clusters(
            pd.read_csv(PROCESSED / "clusters.csv"),
            pd.read_csv(PROCESSED / "products.csv")), failures)

    if "sales" in stages:
        _run(f"series_weekly_{tag}", lambda: validate_series(
            pd.read_parquet(PROCESSED / f"series_weekly_{tag}.parquet")), failures)

    if "features" in stages:
        _run(f"features_{tag}", lambda: validate_features(
            pd.read_parquet(PROCESSED / f"features_{tag}.parquet")), failures)
    return failures


def validate_or_exit(stage: str, tag: str) -> None:
    """Call at the top of a training script: a failed check stops the run."""
    failures = validate_stage(stage, tag)
    if failures:
        print(f"\nDATA VALIDATION FAILED: {failures}. Pipeline stopped.")
        sys.exit(1)
    print("data validation passed\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["raw", "embeddings", "sales", "features", "all"])
    ap.add_argument("--tag", default="cluster_style0.3")
    args = ap.parse_args()
    validate_or_exit(args.stage, args.tag)


if __name__ == "__main__":
    main() 