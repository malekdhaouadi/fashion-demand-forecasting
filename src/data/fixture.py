"""Tiny synthetic dataset so CI can run the whole pipeline without the real data.

Writes products.csv, embeddings.npz, clusters.csv, clusters_model.csv into
data/processed/ and one placeholder image into data/raw/.
It REFUSES to run if products.csv already exists, so it can never overwrite
your real data. Use --force only in a throwaway folder.

Run from the project root:  python -m src.data.fixture
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans

from src.data.load import KEEP

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
SEASONS = ["Summer", "Fall", "Winter", "Spring"]


def build(n_per_type=100, k_true=4, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    image_path = (RAW / "fixture.png").as_posix()
    rows, X, true_cluster = [], [], []
    pid = 1000
    for art in KEEP:
        centers = rng.normal(0, 1, (k_true, dim))
        for i in range(n_per_type):
            c = i % k_true
            rows.append({
                "id": pid, "gender": rng.choice(["Men", "Women"]),
                "masterCategory": "Apparel", "subCategory": "Topwear",
                "articleType": art, "baseColour": rng.choice(["Black", "White", "Blue", "Red"]),
                "season": rng.choice(SEASONS), "year": 2012, "usage": "Casual",
                "productDisplayName": f"fixture {pid}", "image_path": image_path})
            X.append(centers[c] + rng.normal(0, 0.7, dim))
            true_cluster.append(c)
            pid += 1
    products = pd.DataFrame(rows)
    X = np.asarray(X, dtype=np.float32)

    # clusters that define the simulated styles (k=4) ...
    sim = pd.DataFrame({"id": products["id"], "articleType": products["articleType"],
                        "cluster": true_cluster})
    # ... and a different clustering for the model (k=6), as in the real project
    model = sim.copy()
    for art in KEEP:
        m = (products["articleType"] == art).to_numpy()
        model.loc[m, "cluster"] = KMeans(6, n_init=3, random_state=7).fit_predict(X[m])
    for d in (sim, model):
        d["series_id"] = d["articleType"] + "|" + d["cluster"].astype(str)
    return products, X, sim, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if (PROCESSED / "products.csv").exists() and not args.force:
        print("data/processed/products.csv exists: refusing to overwrite real data.")
        sys.exit(1)

    products, X, sim, model = build()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(RAW / "fixture.png")

    products.to_csv(PROCESSED / "products.csv", index=False)
    np.savez_compressed(PROCESSED / "embeddings.npz", ids=products["id"].to_numpy(), X=X)
    sim.to_csv(PROCESSED / "clusters.csv", index=False)
    model.to_csv(PROCESSED / "clusters_model.csv", index=False)
    print(f"fixture: {len(products)} products, X {X.shape}")


if __name__ == "__main__":
    main()