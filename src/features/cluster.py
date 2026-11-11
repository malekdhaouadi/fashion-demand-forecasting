"""PCA + k-means WITHIN each category, plus a grid of example images per cluster.

Run from the project root:  python -m src.features.cluster --k 4
Outputs:
  data/processed/clusters.csv     (id, articleType, cluster, series_id)
  reports/clusters/<category>.png (one row of examples per cluster)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

PROCESSED = Path("data/processed")
REPORTS = Path("reports/clusters")


def cluster_category(Xc: np.ndarray, k: int, n_pca: int, seed: int):
    """PCA then k-means on one category. Returns labels, distance to own centroid."""
    n_comp = min(n_pca, Xc.shape[0] - 1, Xc.shape[1])
    Z = PCA(n_components=n_comp, random_state=seed).fit_transform(Xc)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    sil = silhouette_score(Z, km.labels_)
    return km.labels_, dist, sil


def save_grid(cat, df_cat, labels, dist, k, n_show=8):
    """Row = cluster, columns = the n_show images closest to the centroid."""
    fig, axes = plt.subplots(k, n_show, figsize=(1.6 * n_show, 1.8 * k))
    axes = np.atleast_2d(axes)
    for c in range(k):
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(dist[idx])][:n_show]
        for j in range(n_show):
            ax = axes[c, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(Image.open(df_cat.iloc[idx[j]]["image_path"]))
            if j == 0:
                ax.set_title(f"{cat} / cluster {c}", fontsize=8, loc="left")
    plt.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / f"{cat.replace(' ', '_')}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="clusters per category")
    ap.add_argument("--n-pca", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="clusters.csv")
    args = ap.parse_args()

    products = pd.read_csv(PROCESSED / "products.csv")
    z = np.load(PROCESSED / "embeddings.npz")
    assert (z["ids"] == products["id"].to_numpy()).all(), "ids do not line up"
    X = z["X"]

    out = []
    for cat in sorted(products["articleType"].unique()):
        mask = (products["articleType"] == cat).to_numpy()
        df_cat = products[mask].reset_index(drop=True)
        labels, dist, sil = cluster_category(X[mask], args.k, args.n_pca, args.seed)
        save_grid(cat, df_cat, labels, dist, args.k)

        sizes = np.bincount(labels, minlength=args.k)
        top_colour = [df_cat.loc[labels == c, "baseColour"].value_counts(normalize=True)
                      .iloc[0] for c in range(args.k)]
        print(f"{cat:13s} silhouette={sil:.2f}  sizes={sizes.tolist()}  "
              f"top-colour share={np.round(top_colour, 2).tolist()}")

        out.append(pd.DataFrame({"id": df_cat["id"], "articleType": cat,
                                 "cluster": labels,
                                 "series_id": [f"{cat}|{l}" for l in labels]}))

    res = pd.concat(out, ignore_index=True)
    res.to_csv(PROCESSED / args.out, index=False)
    print(f"saved {len(res)} rows to {PROCESSED / args.out}; grids in {REPORTS}/")


if __name__ == "__main__":
    main()