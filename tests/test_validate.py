import numpy as np
import pandas as pd
import pytest

try:
    import pandera.pandas as pa
except ImportError:
    import pandera as pa

from src.data.load import KEEP
from src.data.validate import validate_products, validate_series

N = 10  # short series, to keep the test fast


def good_series(n_series=2, n_weeks=N):
    rows = []
    for s in range(n_series):
        for w in pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON"):
            rows.append({"series_id": f"Tshirts|{s}", "articleType": "Tshirts",
                         "style_group": s, "week": w, "sales": 5,
                         "price": 20.0, "promo": 0})
    return pd.DataFrame(rows)


def good_products():
    rows = [{"id": i, "gender": "Men", "masterCategory": "Apparel",
             "articleType": a, "baseColour": "Black", "season": "Summer",
             "usage": "Casual", "year": 2012, "image_path": "x.jpg"}
            for i, a in enumerate(KEEP)]
    return pd.DataFrame(rows)


def test_good_series_passes():
    validate_series(good_series(), n_weeks=N)


def test_negative_sales_fail():
    df = good_series()
    df.loc[3, "sales"] = -1
    with pytest.raises(pa.errors.SchemaErrors):
        validate_series(df, n_weeks=N)


def test_bad_promo_fails():
    df = good_series()
    df.loc[0, "promo"] = 2
    with pytest.raises(pa.errors.SchemaErrors):
        validate_series(df, n_weeks=N)


def test_missing_week_fails():
    df = good_series().drop(index=4)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_series(df, n_weeks=N)


def test_duplicate_row_fails():
    df = pd.concat([good_series(), good_series().iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_series(df, n_weeks=N)


def test_good_products_pass():
    validate_products(good_products(), check_images=False, min_per_type=1)


def test_missing_season_fails():
    df = good_products()
    df.loc[0, "season"] = None
    with pytest.raises(pa.errors.SchemaErrors):
        validate_products(df, check_images=False, min_per_type=1)


def test_unknown_category_fails():
    df = good_products()
    df.loc[0, "articleType"] = "Socks"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_products(df, check_images=False, min_per_type=1)