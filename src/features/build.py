"""Build the model table from the series-level weekly sales.

Run from the project root:
    python -m src.features.build --tag cluster_style0.3
    python -m src.features.build --tag cluster_style0.3 --model-clusters clusters_model.csv
Output: data/processed/features_<tag>.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROCESSED = Path("data/processed")
HORIZON = 4                 # forecast horizon used in the backtest
LAGS = [4, 8, 13, 52]
ROLLS = [4, 8, 13]

BASE_FEATURES = (
    ["cat_code", "woy", "woy_sin", "woy_cos",
     "hol_newyear", "hol_summer_sale", "hol_blackfriday", "hol_xmas",
     "price", "promo", "growth", "cat_lag_4", "cat_roll_13"]
    + [f"lag_{L}" for L in LAGS]
    + [f"roll_{w}" for w in ROLLS]
    + [f"season_{s}" for s in ["Summer", "Fall", "Winter", "Spring"]]
)


def visual_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("vis_")]


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every lag is >= HORIZON weeks, so for any target week inside a
    4-week test block, the feature was already known at the forecast origin."""
    df = df.sort_values(["series_id", "week"]).reset_index(drop=True)
    g = df.groupby("series_id")["sales"]

    for L in LAGS:
        df[f"lag_{L}"] = g.shift(L)

    past = g.shift(HORIZON)
    for w in ROLLS:
        df[f"roll_{w}"] = (past.groupby(df["series_id"]).rolling(w).mean()
                           .reset_index(level=0, drop=True))

    # recent level vs the same period a year earlier (captures a trend)
    df["growth"] = (df["roll_13"] / df.groupby("series_id")["roll_13"].shift(52))
    df["growth"] = df["growth"].replace([np.inf, -np.inf], np.nan)

    # category-level history (total over its style groups)
    cat = (df.groupby(["articleType", "week"], as_index=False)["sales"].sum()
             .rename(columns={"sales": "cat_sales"})
             .sort_values(["articleType", "week"]))
    cg = cat.groupby("articleType")["cat_sales"]
    cat["cat_lag_4"] = cg.shift(HORIZON)
    cat["cat_roll_13"] = (cg.shift(HORIZON).groupby(cat["articleType"])
                          .rolling(13).mean().reset_index(level=0, drop=True))
    return df.merge(cat[["articleType", "week", "cat_lag_4", "cat_roll_13"]],
                    on=["articleType", "week"], how="left")


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    woy = df["week"].dt.isocalendar().week.astype(int)
    df["woy"] = woy
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52)
    df["hol_newyear"] = woy.isin([1, 2, 3]).astype(int)
    df["hol_summer_sale"] = woy.isin([27, 28]).astype(int)
    df["hol_blackfriday"] = (woy == 47).astype(int)
    df["hol_xmas"] = woy.isin([50, 51]).astype(int)
    df["cat_code"] = df["articleType"].astype("category").cat.codes
    return df


def static_features(sim_file: str, model_file: str, n_pca: int) -> pd.DataFrame:
    """One row per series: season mix (a real label) and visual features.
    No series-size feature: it would be a proxy for sales volume, not vision."""
    products = pd.read_csv(PROCESSED / "products.csv")
    sim = pd.read_csv(PROCESSED / sim_file)[["id", "cluster"]]
    mod = pd.read_csv(PROCESSED / model_file)[["id", "cluster"]]
    p = (products.merge(sim.rename(columns={"cluster": "sim_cluster"}), on="id")
                 .merge(mod.rename(columns={"cluster": "model_cluster"}), on="id"))
    # same definition of a series as in simulate.py
    p["series_id"] = p["articleType"] + "|" + p["sim_cluster"].astype(str)

    # season mix: not visual, it is the product's season label
    season = pd.crosstab(p["series_id"], p["season"], normalize="index")
    season = season.reindex(columns=["Summer", "Fall", "Winter", "Spring"],
                            fill_value=0).add_prefix("season_")

    # visual: which model-clusters the series' products belong to
    share = pd.crosstab(p["series_id"], p["model_cluster"],
                        normalize="index").add_prefix("vis_share_c")
    majority = (p.groupby("series_id")["model_cluster"]
                 .agg(lambda s: s.value_counts().idxmax())
                 .rename("vis_majority"))

    # visual: mean of the first PCA components of the members' embeddings
    z = np.load(PROCESSED / "embeddings.npz")
    assert (z["ids"] == products["id"].to_numpy()).all(), "ids do not line up"
    pcs = PCA(n_components=n_pca, random_state=42).fit_transform(z["X"])
    pc = pd.DataFrame(pcs, columns=[f"vis_pca{i}" for i in range(n_pca)])
    pc["id"] = products["id"].to_numpy()
    pc = (pc.merge(p[["id", "series_id"]], on="id")
            .groupby("series_id").mean(numeric_only=True))
    pc = pc[[c for c in pc.columns if c.startswith("vis_pca")]]

    return pd.concat([season, share, majority, pc], axis=1).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cluster_style0.3")
    ap.add_argument("--sim-clusters", default="clusters.csv",
                    help="clusters that DEFINE the series (same as the simulation)")
    ap.add_argument("--model-clusters", default=None,
                    help="clusters the MODEL sees; default = same as simulation")
    ap.add_argument("--n-pca", type=int, default=5)
    args = ap.parse_args()
    model_file = args.model_clusters or args.sim_clusters

    df = pd.read_parquet(PROCESSED / f"series_weekly_{args.tag}.parquet")
    df = history_features(df)
    df = calendar_features(df)
    df = df.merge(static_features(args.sim_clusters, model_file, args.n_pca),
                  on="series_id", how="left")

    out = PROCESSED / f"features_{args.tag}.parquet"
    df.to_parquet(out, index=False)

    vis = visual_features(df)
    print(f"saved {df.shape} to {out}")
    print(f"base features: {len(BASE_FEATURES)} | visual features: {len(vis)}")
    print("visual:", vis)
    print("missing base features:", [c for c in BASE_FEATURES if c not in df.columns])
    print("NaN share per history feature:")
    print(df[["lag_4", "lag_52", "roll_13", "growth", "cat_roll_13"]]
          .isna().mean().round(2))


if __name__ == "__main__":
    main()