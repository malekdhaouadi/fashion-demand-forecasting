from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app, future_features, load_bundle

BUNDLE = Path("bundle")
pytestmark = pytest.mark.skipif(not (BUNDLE / "meta.json").exists(),
                                reason="no bundle (run python -m src.serving.export)")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("PRED_LOG", str(tmp_path_factory.mktemp("logs") / "pred.jsonl"))
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_no_train_serve_skew():
    """Features rebuilt from truncated history == features computed in training."""
    b = load_bundle(BUNDLE)
    weeks = np.sort(b["history"]["week"].unique())
    target = pd.Timestamp(weeks[120])
    hist = b["history"][b["history"]["week"] < target]
    b2 = {**b, "history": hist, "last_week": hist["week"].max()}
    cat = b["history"]["articleType"].iloc[0]

    got = future_features(b2, target, cat, promo=0).sort_values("series_id")
    ref = b["table"][b["table"]["week"] == target].sort_values("series_id")
    cols = [c for c in b["feature_names"] if c not in ("price", "promo")]
    assert np.allclose(got[cols].astype(float), ref[cols].astype(float),
                       equal_nan=True, atol=1e-9)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_future_forecast_is_valid(client, h):
    last = pd.Timestamp(client.get("/health").json()["last_history_week"])
    cat = next(iter(client.get("/series").json()))
    r = client.post("/forecast", json={"category": cat,
                                       "week": str((last + pd.Timedelta(weeks=h)).date())})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "future" and body["forecasts"]
    for f in body["forecasts"]:
        assert 0 <= f["q10"] <= f["q50"] <= f["q67"] <= f["q90"]


def test_too_far_ahead_is_rejected(client):
    last = pd.Timestamp(client.get("/health").json()["last_history_week"])
    cat = next(iter(client.get("/series").json()))
    r = client.post("/forecast", json={"category": cat,
                                       "week": str((last + pd.Timedelta(weeks=5)).date())})
    assert r.status_code == 422


def test_unknown_category(client):
    r = client.post("/forecast", json={"category": "Socks", "week": "2023-06-05"})
    assert r.status_code == 404


def test_historical_week_is_flagged(client):
    cat = next(iter(client.get("/series").json()))
    r = client.post("/forecast", json={"category": cat, "week": "2022-06-06"})
    assert r.status_code == 200
    assert r.json()["mode"] == "historical" and r.json()["in_training_window"] is True