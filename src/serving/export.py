"""Export the PRODUCTION models and a data snapshot into a self-contained bundle.

Run from the project root:  python -m src.serving.export --tag cluster_style0.3
Output: bundle/ (model_q10.json ... model_q90.json, features.parquet, meta.json)
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src import tracking
from src.models.promote import load_alias
from src.models.xgb import QUANTILES

PROCESSED = Path("data/processed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cluster_style0.3")
    ap.add_argument("--out", default="bundle")
    args = ap.parse_args()

    mlflow.set_tracking_uri(tracking.TRACKING_URI)
    prod = load_alias(MlflowClient(), "production")
    if prod is None:
        print("no model with alias 'production' in the registry; run promote.py first")
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for q, m in prod["models"].items():
        m.save_model(str(out / f"model_q{int(q * 100)}.json"))
    shutil.copy(PROCESSED / f"features_{args.tag}.parquet", out / "features.parquet")

    meta = {"model_version": prod["version"], "trained_until": prod["trained_until"],
            "tag": args.tag, "features": prod["features"], "quantiles": QUANTILES,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"bundle written to {out}/ (production v{prod['version']}, "
          f"trained until {prod['trained_until']}, {len(prod['features'])} features)")


if __name__ == "__main__":
    main()