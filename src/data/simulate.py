"""Simulate weekly sales per product.

Images and labels are REAL (Fashion Product Images). Sales are SIMULATED.

Run from the project root:
    python -m src.data.simulate --style-effect 0.3
    python -m src.data.simulate --style-effect 0.0   # null experiment
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
N_WEEKS = 156                      # 3 years
START = "2021-01-04"               # a Monday
SEED = 42

# articleType -> (mean weekly units per product, base price in EUR)
CATEGORY = {
    "Tshirts": (2.0, 20), "Shirts": (1.5, 35), "Tops": (1.5, 25),
    "Jeans": (1.5, 50), "Shorts": (1.2, 30), "Casual Shoes": (1.2, 60),
    "Sports Shoes": (1.2, 80), "Heels": (0.8, 55), "Handbags": (0.8, 70),
    "Watches": (0.6, 90),
}

# week of the year at which each season peaks
SEASON_PEAK = {"Summer": 27, "Fall": 40, "Winter": 50, "Spring": 15}
SEASON_AMPLITUDE = 0.4

# week of year -> demand multiplier
HOLIDAYS = {1: 1.3, 2: 1.3, 3: 1.2, 27: 1.2, 28: 1.2,
            47: 1.6, 50: 1.4, 51: 1.5}

PROMO_PROB = 0.10                  # chance a category is on promo in a week
PROMO_DISCOUNT = 0.20
PRICE_ELASTICITY = -1.5


def simulate(products: pd.DataFrame, style_col: str, style_effect: float,
             seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    style_rng = np.random.default_rng(seed + 1)   # separate: on/off changes nothing else

    n = len(products)
    cat = products["articleType"].to_numpy()
    weeks = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    t = np.arange(N_WEEKS)

    # per-product base level and price
    level = np.array([CATEGORY[c][0] for c in cat]) * rng.lognormal(0, 0.5, n)
    price0 = np.array([CATEGORY[c][1] for c in cat]) * rng.lognormal(0, 0.25, n)

    # seasonality from the product's season label
    peak = products["season"].map(SEASON_PEAK).fillna(27).to_numpy()
    seas = 1 + SEASON_AMPLITUDE * np.cos(
        2 * np.pi * (woy[None, :] - peak[:, None]) / 52)

    # holidays (same for everyone)
    hol = np.ones(N_WEEKS)
    for w, m in HOLIDAYS.items():
        hol[woy == w] = m

    # promotions: shared by all products of a category
    promo_cat = {c: rng.random(N_WEEKS) < PROMO_PROB for c in sorted(set(cat))}
    promo = np.vstack([promo_cat[c] for c in cat])
    discount = promo * PROMO_DISCOUNT
    price = price0[:, None] * (1 - discount)
    price_mult = (1 - discount) ** PRICE_ELASTICITY

    # style trend: each (category, style group) gets its own yearly slope
    groups = products[style_col].to_numpy()
    slopes = {key: style_rng.uniform(-1, 1) * style_effect
              for key in sorted(set(zip(cat, groups)))}
    slope = np.array([slopes[(c, g)] for c, g in zip(cat, groups)])
    style_mult = np.exp(slope[:, None] * t[None, :] / 52)

    # expected demand, then noise
    mu = level[:, None] * seas * hol[None, :] * price_mult * style_mult
    noise = rng.gamma(shape=20, scale=1 / 20, size=mu.shape)
    sales = rng.poisson(mu * noise)

    return pd.DataFrame({
        "product_id": np.repeat(products["id"].to_numpy(), N_WEEKS),
        "articleType": np.repeat(cat, N_WEEKS),
        "style_group": np.repeat(groups, N_WEEKS),
        "week": np.tile(weeks, n),
        "sales": sales.ravel(),
        "price": price.ravel().round(2),
        "promo": promo.ravel().astype(int),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style-effect", type=float, default=0.3,
                    help="max yearly log-growth of a style group; 0 = off")
    ap.add_argument("--style-col", default="baseColour",
                    help="TEMPORARY stand-in for the visual cluster id")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    products = pd.read_csv(PROCESSED / "products.csv")
    sales = simulate(products, args.style_col, args.style_effect, args.seed)

    tag = f"style{args.style_effect}"
    sales.to_parquet(PROCESSED / f"sales_{tag}.parquet", index=False)

    weekly = (sales.groupby(["articleType", "week"], as_index=False)
                   .agg(sales=("sales", "sum"), price=("price", "mean"),
                        promo=("promo", "max")))
    weekly.to_parquet(PROCESSED / f"category_weekly_{tag}.parquet", index=False)

    print(f"{len(sales):,} product-week rows -> sales_{tag}.parquet")
    print(weekly.groupby("articleType")["sales"].mean().round(0))


if __name__ == "__main__":
    main()