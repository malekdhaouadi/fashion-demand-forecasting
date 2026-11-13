"""Cold-start experiment: forecast NEW products that have no sales history.

Held-out products are never seen by the model. The four variants differ only
in WHO a new product borrows history from (see VARIANTS).

Run from the project root:
    python -m src.models.coldstart --tag cluster_style0.3
    python -m src.models.coldstart --tag cluster_style0.0
Output: data/processed/coldstart_preds_<tag>.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import mlflow

from src import tracking

from src.models.evaluate import rolling_origin

PROCESSED = Path("data/processed")
HORIZON = 4
T_MIN = 16                      # first week with a full 13-week lagged window
SEASONS = ["Summer", "Fall", "Winter", "Spring"]
VARIANTS = ["category", "random_k", "visual_knn", "oracle_cluster"]
PARAMS = dict(n_estimators=250, learning_rate=0.05, max_depth=5, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, random_state=42,
              n_jobs=-1, tree_method="hist")


# ---------- data ----------
def load(tag):
    products = pd.read_csv(PROCESSED / "products.csv")
    ids = products["id"].to_numpy()
    sales = pd.read_parquet(PROCESSED / f"sales_{tag}.parquet")

    def mat(col):                                   # (products x weeks)
        return (sales.pivot(index="product_id", columns="week", values=col)
                     .loc[ids].to_numpy(dtype=float))

    weeks = pd.DatetimeIndex(np.sort(sales["week"].unique()))
    sim = sales.groupby("product_id")["style_group"].first().loc[ids].to_numpy()
    z = np.load(PROCESSED / "embeddings.npz")
    assert (z["ids"] == ids).all(), "ids do not line up"
    Xn = z["X"] / np.linalg.norm(z["X"], axis=1, keepdims=True)
    return products, mat("sales"), mat("price"), mat("promo"), weeks, sim, Xn


def history_mats(S):
    """Lagged history matrices. Week t only uses sales up to t - HORIZON."""
    past = np.full_like(S, np.nan)
    past[:, HORIZON:] = S[:, :-HORIZON]
    roll13 = pd.DataFrame(past.T).rolling(13).mean().to_numpy().T
    lag52 = np.full_like(S, np.nan)
    lag52[:, 52:] = S[:, :-52]
    roll13_52 = np.full_like(S, np.nan)
    roll13_52[:, 52:] = roll13[:, :-52]
    return roll13, lag52, roll13_52


# ---------- who to borrow from ----------
def split_products(series, frac, rng):
    """Hold out `frac` of the products of every style group (never seen)."""
    known = np.ones(len(series), dtype=bool)
    for g in np.unique(series):
        idx = np.where(series == g)[0]
        n_new = max(1, int(round(frac * len(idx))))
        known[rng.choice(idx, n_new, replace=False)] = False
    return known


def group_mean(M, groups, known):
    """Mean over KNOWN products of the same group (self excluded for known)."""
    out = np.full(M.shape, np.nan)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        kn = idx[known[idx]]
        assert len(kn) >= 2, f"group {g} has too few known products"
        s = M[kn].sum(axis=0)
        out[idx] = s / len(kn)
        out[kn] = (s[None, :] - M[kn]) / (len(kn) - 1)
    return out


def index_mean(M, NB, chunk=500):
    out = np.empty_like(M)
    for s in range(0, len(M), chunk):
        out[s:s + chunk] = M[NB[s:s + chunk]].mean(axis=1)
    return out


def knn_neighbours(Xn, cats, known, k):
    """K visually nearest KNOWN products in the same category (cosine)."""
    NB = np.zeros((len(Xn), k), dtype=int)
    for c in np.unique(cats):
        idx = np.where(cats == c)[0]
        kn = idx[known[idx]]
        sim = Xn[idx] @ Xn[kn].T
        sim[idx[:, None] == kn[None, :]] = -np.inf          # no self-neighbour
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        NB[idx] = kn[top]
    return NB


def random_neighbours(cats, known, k, rng):
    NB = np.zeros((len(cats), k), dtype=int)
    for c in np.unique(cats):
        idx = np.where(cats == c)[0]
        kn = idx[known[idx]]
        for i in idx:
            NB[i] = rng.choice(kn[kn != i], size=k, replace=False)
    return NB


def nbr_feats(mean_fn, hist):
    R, L, R5 = (mean_fn(m) for m in hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = R / R5
    g[~np.isfinite(g)] = np.nan
    return {"nbr_roll_13": R, "nbr_lag_52": L, "nbr_growth": g}


# ---------- model ----------
def frame(i, t, static, timev, mats):
    cols = {k: v[i] for k, v in static.items()}
    cols.update({k: v[t] for k, v in timev.items()})
    cols.update({k: v[i, t] for k, v in mats.items()})
    return pd.DataFrame(cols)


def score(d):
    """Series-level WAPE: sum the new products of each style group per week,
    then compare. Product-level counts are too noisy to show a style effect."""
    agg = d.groupby(["series", "week"])[["y", "yhat"]].sum()
    return {"WAPE": float((agg.y - agg.yhat).abs().sum() / agg.y.sum()),
            "Bias": float(agg.yhat.sum() / agg.y.sum() - 1),
            "RMSE_product": float(np.sqrt(((d.y - d.yhat) ** 2).mean()))}


def run_split(split, args, data):
    products, S, PRICE, PROMO, weeks, sim, Xn = data
    N, T = S.shape
    cats = products["articleType"].to_numpy()
    series = np.array([f"{c}|{s}" for c, s in zip(cats, sim)])
    rng = np.random.default_rng(1000 + split)

    known = split_products(series, args.holdout, rng)
    held, pool = np.where(~known)[0], np.where(known)[0]
    train_prod = pool
    if args.max_train_products and len(pool) > args.max_train_products:
        train_prod = rng.choice(pool, args.max_train_products, replace=False)

    hist = history_mats(S)
    NB_knn = knn_neighbours(Xn, cats, known, args.k)
    NB_rnd = random_neighbours(cats, known, args.k, rng)
    purity = lambda NB: float((sim[NB[held]] == sim[held][:, None]).mean())
    print(f"split {split}: {len(held)} new products | neighbours in the same "
          f"true style group: visual {purity(NB_knn):.2f} vs random {purity(NB_rnd):.2f}")

    mean_fns = {
        "category": lambda M: group_mean(M, cats, known),
        "random_k": lambda M: index_mean(M, NB_rnd),
        "visual_knn": lambda M: index_mean(M, NB_knn),
        "oracle_cluster": lambda M: group_mean(M, series, known),
    }
    nbr = {v: nbr_feats(fn, hist) for v, fn in mean_fns.items()}

    woy = weeks.isocalendar().week.to_numpy().astype(int)
    timev = {"woy": woy, "woy_sin": np.sin(2 * np.pi * woy / 52),
             "woy_cos": np.cos(2 * np.pi * woy / 52),
             "hol_newyear": np.isin(woy, [1, 2, 3]).astype(int),
             "hol_summer_sale": np.isin(woy, [27, 28]).astype(int),
             "hol_blackfriday": (woy == 47).astype(int),
             "hol_xmas": np.isin(woy, [50, 51]).astype(int)}
    static = {"cat_code": pd.Categorical(cats).codes.astype(int)}
    for s in SEASONS:                                   # catalogue label: known at launch
        static[f"season_{s}"] = (products["season"].to_numpy() == s).astype(int)

    pca = {}
    if args.with_pca:                                   # fit on known products only
        P = PCA(n_components=5, random_state=42).fit(Xn[known]).transform(Xn)
        pca = {f"vis_pca{j}": P[:, j] for j in range(5)}

    def spec(v):
        st = dict(static)
        if v == "visual_knn":
            st.update(pca)
        return st, {"price": PRICE, "promo": PROMO, **nbr[v]}

    folds = list(rolling_origin(T, args.initial, args.horizon, args.step))
    out = []
    for fold, (a, b) in enumerate(folds):
        ts = np.arange(T_MIN, a)                        # train strictly before the origin
        i_tr, t_tr = np.repeat(train_prod, len(ts)), np.tile(ts, len(train_prod))
        tb = np.arange(a, b)
        i_te, t_te = np.repeat(held, len(tb)), np.tile(tb, len(held))
        for v in VARIANTS:
            st, mt = spec(v)
            m = XGBRegressor(objective="count:poisson", **PARAMS)
            m.fit(frame(i_tr, t_tr, st, timev, mt), S[i_tr, t_tr])
            yhat = m.predict(frame(i_te, t_te, st, timev, mt))
            out.append(pd.DataFrame({
                "variant": v, "split": split, "fold": fold, "product": i_te,
                "series": series[i_te], "week": t_te,
                "y": S[i_te, t_te], "yhat": yhat}))
        print(f"  fold {fold + 1}/{len(folds)}", end="\r")
    print()
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cluster_style0.3")
    ap.add_argument("--holdout", type=float, default=0.2, help="share of new products")
    ap.add_argument("--k", type=int, default=20, help="neighbours per product")
    ap.add_argument("--n-splits", type=int, default=1, help="different random holdouts")
    ap.add_argument("--max-train-products", type=int, default=2000,
                    help="speed: train on a sample of known products (0 = all)")
    ap.add_argument("--with-pca", action="store_true",
                    help="also give the visual variant 5 PCA components")
    ap.add_argument("--initial", type=int, default=104)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--step", type=int, default=4)
    args = ap.parse_args()

    data = load(args.tag)
    res = pd.concat([run_split(s, args, data) for s in range(args.n_splits)],
                    ignore_index=True)
    preds_file = PROCESSED / f"coldstart_preds_{args.tag}.parquet"
    res.to_parquet(preds_file, index=False)

    pf = pd.DataFrame([{"variant": v, "split": s, "fold": f, **score(g)}
                       for (v, s, f), g in res.groupby(["variant", "split", "fold"])])

    print(f"\n=== cold start | {args.tag} | {len(pf) // len(VARIANTS)} fold-runs ===")
    for v in VARIANTS:
        d = pf[pf.variant == v]
        print(f"{v:15s} WAPE {100 * d.WAPE.mean():5.1f}% +/- {100 * d.WAPE.std():4.1f}   "
              f"bias {100 * d.Bias.mean():+5.1f}%   RMSE/product {d.RMSE_product.mean():.3f}")

    w = pf.pivot(index=["split", "fold"], columns="variant", values="WAPE")
    print("\nPaired WAPE difference in points (negative = first is better):")
    paired = {}
    for x, y in [("random_k", "category"), ("visual_knn", "category"),
                 ("visual_knn", "random_k"), ("oracle_cluster", "visual_knn")]:
        d = 100 * (w[x] - w[y])
        paired[f"paired_wape_{x}_minus_{y}"] = d.mean()
        print(f"  {x} - {y}: mean {d.mean():+.2f}  std {d.std():.2f}  "
              f"lower in {(d < 0).sum()}/{len(d)}")

    # ---------- MLflow ----------
    params = {"tag": args.tag, "holdout": args.holdout, "k": args.k,
              "n_splits": args.n_splits, "max_train_products": args.max_train_products,
              "with_pca": args.with_pca, "initial": args.initial,
              "horizon": args.horizon, "step": args.step,
              **{f"xgb.{k}": v for k, v in PARAMS.items()}}
    data_files = [PROCESSED / f"sales_{args.tag}.parquet",
                  PROCESSED / "products.csv", PROCESSED / "embeddings.npz"]
    with tracking.run("fashion-coldstart", f"coldstart_{args.tag}",
                      params=params, data_files=data_files):
        tracking.log_metrics(paired)
        mlflow.log_artifact(str(preds_file))
        for v in VARIANTS:
            d = pf[pf.variant == v]
            with tracking.run("fashion-coldstart", v, nested=True, tags={"variant": v}):
                tracking.log_metrics({
                    "WAPE_mean": d.WAPE.mean(), "WAPE_std": d.WAPE.std(),
                    "Bias_mean": d.Bias.mean(),
                    "RMSE_product_mean": d.RMSE_product.mean()})
                for fold, wape in d.groupby("fold").WAPE.mean().items():
                    tracking.log_metrics({"fold_WAPE": wape}, step=int(fold))
    print("\nlogged to MLflow")


if __name__ == "__main__":
    main()