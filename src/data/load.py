"""Build the product subset used by the whole project.

Run from the project root:  python -m src.data.load
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

KEEP = ["Tshirts", "Shirts", "Tops", "Jeans", "Shorts",
        "Casual Shoes", "Sports Shoes", "Heels", "Handbags", "Watches"]
MAX_PER_TYPE = 500
SEED = 42


def load_styles() -> pd.DataFrame:
    # on_bad_lines="skip": a few rows of styles.csv are malformed
    return pd.read_csv(RAW / "styles.csv", on_bad_lines="skip")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["baseColour", "season", "usage", "year"]).copy()
    print(f"dropped {n0 - len(df)} rows with missing labels")

    df = df[df["articleType"].isin(KEEP)].copy()

    has_image = df["id"].apply(lambda i: (RAW / "images" / f"{i}.jpg").exists())
    print(f"dropped {(~has_image).sum()} rows without an image file")
    df = df[has_image].copy()

    df["image_path"] = df["id"].apply(lambda i: str(RAW / "images" / f"{i}.jpg"))
    return df


def sample_per_type(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("articleType"):
        parts.append(g.sample(min(len(g), MAX_PER_TYPE), random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def main() -> None:
    df = clean(load_styles())
    subset = sample_per_type(df)

    missing_types = set(KEEP) - set(subset["articleType"])
    if missing_types:
        print("WARNING: no rows for", missing_types)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "products.csv"
    subset.to_csv(out, index=False)

    print(subset["articleType"].value_counts())
    print(f"saved {len(subset)} products to {out}")


if __name__ == "__main__":
    main()