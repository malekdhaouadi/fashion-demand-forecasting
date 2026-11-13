"""Thin MLflow helper: one place for the tracking URI, data versions and git info.

Every run records WHAT data it saw (content hash of each input file) and WHICH
code produced it (git commit). DVC will replace the hashes in week 4.
"""
import hashlib
import math
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()


def file_hash(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_info() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              cwd=ROOT, check=True).stdout.strip()
    try:
        return {"git_commit": git("rev-parse", "--short", "HEAD"),
                "git_dirty": str(bool(git("status", "--porcelain")))}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def _ensure_experiment(name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(name)


@contextmanager
def run(experiment, run_name, params=None, data_files=(), tags=None, nested=False):
    """Start a run (or a child run) with params, data hashes and git info."""
    if nested:
        mlflow.set_tracking_uri(TRACKING_URI)
    else:
        _ensure_experiment(experiment)
    with mlflow.start_run(run_name=run_name, nested=nested) as r:
        mlflow.set_tags({**git_info(),
                         **{f"data.{Path(p).name}": file_hash(Path(p)) for p in data_files},
                         **(tags or {})})
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        yield r


def log_metrics(d: dict, step=None) -> None:
    """Log a dict of metrics, skipping None and NaN."""
    clean = {k: float(v) for k, v in d.items()
             if v is not None and not math.isnan(float(v))}
    if clean:
        mlflow.log_metrics(clean, step=step)