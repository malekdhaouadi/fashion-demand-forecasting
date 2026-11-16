"""FastAPI forecast service. Reads only the exported bundle (no MLflow, no data/).

Run from the project root:
    python -m uvicorn src.serving.app:app --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import json
import os
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBRegressor

from src.data.simulate import PROMO_DISCOUNT
from src.features.build import HORIZON, calendar_features, history_features

QUANTILES = [0.1, 0.5, 0.67, 0.9]
BASE_COLS = ["series_id", "articleType", "style_group", "week", "sales", "price", "promo"]
_LOG_LOCK = threading.Lock()


# ---------------------------------------------------------------- bundle
def load_bundle(path: Path) -> dict:
    meta = json.loads((path / "meta.json").read_text())
    table = pd.read_parquet(path / "features.parquet")
    models = {}
    for q in QUANTILES:
        m = XGBRegressor()
        m.load_model(str(path / f"model_q{int(q * 100)}.json"))
        models[q] = m
    static_cols = [c for c in table.columns if c.startswith(("season_", "vis_"))]
    history = table[BASE_COLS].copy()
    return {
        "meta": meta, "models": models, "table": table, "history": history,
        "static": table[["series_id", *static_cols]].drop_duplicates("series_id"),
        "feature_names": meta["features"],
        "first_week": history["week"].min(), "last_week": history["week"].max(),
    }


def predict(b: dict, X: pd.DataFrame) -> np.ndarray:
    """Models were trained on log1p(sales). Sorting prevents crossing quantiles."""
    P = np.column_stack([np.expm1(b["models"][q].predict(X)).clip(min=0) for q in QUANTILES])
    return np.sort(P, axis=1)


def future_features(b: dict, target: pd.Timestamp, category: str, promo: int) -> pd.DataFrame:
    """Rows of every series for a week after the last observed one, built with the
    same functions as in training. Only the target week carries price/promo."""
    hist, last = b["history"], b["last_week"]
    n = (target - last).days // 7
    template = hist.loc[hist["week"] == last, ["series_id", "articleType", "style_group"]]
    base_price = (hist[hist["promo"] == 0].sort_values("week")
                  .groupby("series_id")["price"].last())
    new = []
    for k in range(1, n + 1):
        d = template.copy()
        d["week"] = last + pd.Timedelta(weeks=k)
        d["sales"] = np.nan                      # unknown; never used by a lag >= 4 weeks
        d["promo"] = int(k == n and promo == 1) * (d["articleType"] == category).astype(int)
        d["price"] = d["series_id"].map(base_price) * (1 - PROMO_DISCOUNT * d["promo"])
        new.append(d)
    df = pd.concat([hist, *new], ignore_index=True)
    df = calendar_features(history_features(df))
    df = df.merge(b["static"], on="series_id", how="left")
    return df[df["week"] == target]


def log_prediction(record: dict) -> None:
    """One JSON line per request, so live error can be computed once actuals arrive."""
    path = Path(os.environ.get("PRED_LOG", "logs/predictions.jsonl"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass                                     # logging must never break serving


# ---------------------------------------------------------------- API
@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("BUNDLE_DIR", "bundle"))
    if not (path / "meta.json").exists():
        raise RuntimeError(f"no model bundle in '{path}'. Run: python -m src.serving.export")
    app.state.bundle = load_bundle(path)
    yield


app = FastAPI(title="Fashion demand forecast", version="0.1.0", lifespan=lifespan)


class ForecastRequest(BaseModel):
    category: str
    week: date
    style_group: int | None = None
    promo: int = Field(0, ge=0, le=1, description="1 = category on promotion that week")


class SeriesForecast(BaseModel):
    series_id: str
    style_group: int
    q10: float
    q50: float
    q67: float = Field(description="suggested stock level (cost-optimal quantile)")
    q90: float


class ForecastResponse(BaseModel):
    category: str
    week: date
    mode: str
    in_training_window: bool
    model_version: int
    trained_until: str
    forecasts: list[SeriesForecast]


@app.get("/health")
def health():
    b = getattr(app.state, "bundle", None)
    if b is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ok", "model_version": b["meta"]["model_version"],
            "trained_until": b["meta"]["trained_until"],
            "last_history_week": str(b["last_week"].date()),
            "max_forecast_week": str((b["last_week"] + pd.Timedelta(weeks=HORIZON)).date())}


@app.get("/series")
def series():
    h = app.state.bundle["history"].drop_duplicates("series_id")
    return {c: sorted(int(x) for x in g["style_group"]) for c, g in h.groupby("articleType")}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    b = app.state.bundle
    ts = pd.Timestamp(req.week)
    target = ts - pd.Timedelta(days=ts.weekday())          # weeks start on Monday
    first, last = b["first_week"], b["last_week"]

    if req.category not in set(b["history"]["articleType"]):
        raise HTTPException(404, f"unknown category '{req.category}'; see GET /series")
    if target < first:
        raise HTTPException(422, f"week is before the data start ({first.date()})")
    if target > last + pd.Timedelta(weeks=HORIZON):
        raise HTTPException(422, f"week {target.date()} is too far ahead: features need sales "
                                 f"history, so the last forecastable week is "
                                 f"{(last + pd.Timedelta(weeks=HORIZON)).date()}")

    if target <= last:
        rows, mode = b["table"][b["table"]["week"] == target], "historical"
    else:
        rows, mode = future_features(b, target, req.category, req.promo), "future"
    rows = rows[rows["articleType"] == req.category]
    if req.style_group is not None:
        rows = rows[rows["style_group"] == req.style_group]
    if rows.empty:
        raise HTTPException(404, "no such series; see GET /series")

    P = predict(b, rows[b["feature_names"]].astype(float))
    out = [SeriesForecast(series_id=r.series_id, style_group=int(r.style_group),
                          q10=round(float(p[0]), 2), q50=round(float(p[1]), 2),
                          q67=round(float(p[2]), 2), q90=round(float(p[3]), 2))
           for r, p in zip(rows.itertuples(), P)]

    trained_until = b["meta"]["trained_until"]
    for f in out:
        log_prediction({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "series_id": f.series_id, "week": str(target.date()), "mode": mode,
                        "q10": f.q10, "q50": f.q50, "q67": f.q67, "q90": f.q90,
                        "model_version": b["meta"]["model_version"]})
    return ForecastResponse(category=req.category, week=target.date(), mode=mode,
                            in_training_window=bool(target < pd.Timestamp(trained_until)),
                            model_version=b["meta"]["model_version"],
                            trained_until=trained_until, forecasts=out)