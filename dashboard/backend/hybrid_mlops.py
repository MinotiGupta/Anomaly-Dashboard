"""Local model lifecycle and run tracking for the physics + Isolation Forest model."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest


MODEL_NAME = "physics-constrained-isolation-forest"
MODEL_VERSION = "1.0.0"
MODEL_PARAMETERS = {
    "n_estimators": 100,
    "contamination": 0.01,
    "random_state": 42,
    "n_jobs": 1,
}
FEATURE_NAMES = ["temperature", "dT_dt_raw"]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as temp_file:
        json.dump(content, temp_file, indent=2, sort_keys=True)
        temp_path = Path(temp_file.name)
    os.replace(temp_path, path)


def _record_run(database_path: str | Path, metadata: dict[str, Any]) -> None:
    database_path = str(database_path)
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS mlops_model_runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO mlops_model_runs
                (run_id, model_name, model_version, status, created_at,
                 artifact_path, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["run_id"],
                metadata["model_name"],
                metadata["model_version"],
                metadata["status"],
                metadata["created_at"],
                metadata["artifact_path"],
                json.dumps(metadata, separators=(",", ":")),
            ),
        )


def get_tracked_runs(database_path: str | Path, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent durable model run records."""
    if not Path(database_path).exists():
        return []
    with sqlite3.connect(str(database_path)) as connection:
        rows = connection.execute(
            "SELECT metadata_json FROM mlops_model_runs ORDER BY created_at DESC LIMIT ?",
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def get_tracked_run(database_path: str | Path, run_id: str) -> dict[str, Any] | None:
    """Return one durable run record, if it exists."""
    if not Path(database_path).exists():
        return None
    with sqlite3.connect(str(database_path)) as connection:
        row = connection.execute(
            "SELECT metadata_json FROM mlops_model_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def train_and_register(
    *,
    temperatures: np.ndarray,
    rates: np.ndarray,
    run_id: str,
    source_path: str | Path,
    artifact_root: str | Path,
    tracking_database: str | Path,
    physics_veto_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit, validate, persist, and track a reproducible per-upload model run."""
    started_at = datetime.now(timezone.utc).isoformat()
    fit_started = time.perf_counter()
    features = np.column_stack((temperatures, rates)).astype(np.float64, copy=False)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Isolation Forest expects temperature and dT/dt features")
    if features.shape[0] < 2:
        raise ValueError("Isolation Forest requires at least two valid data points")
    if not np.isfinite(features).all():
        raise ValueError("Model features contain non-finite values")

    model = IsolationForest(**MODEL_PARAMETERS)
    model.fit(features)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    predictions = model.predict(features)
    scores = model.decision_function(features)
    predict_seconds = time.perf_counter() - predict_started
    if predictions.shape[0] != features.shape[0] or not np.isfinite(scores).all():
        raise ValueError("Model validation failed: output does not match input rows")

    outlier_mask = predictions == -1
    run_artifact_dir = Path(artifact_root) / f"{MODEL_NAME}_{MODEL_VERSION}" / run_id
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run_artifact_dir / "model.joblib"
    with tempfile.NamedTemporaryFile(dir=run_artifact_dir, delete=False, suffix=".joblib.tmp") as temp_file:
        temp_artifact = Path(temp_file.name)
    try:
        joblib.dump(model, temp_artifact)
        os.replace(temp_artifact, artifact_path)
    finally:
        if temp_artifact.exists():
            temp_artifact.unlink()
    registered_model = joblib.load(artifact_path)
    registered_predictions = registered_model.predict(features)
    registered_scores = registered_model.decision_function(features)
    if not np.array_equal(predictions, registered_predictions) or not np.allclose(scores, registered_scores):
        raise ValueError("Registered artifact failed prediction parity validation")

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "model_id": f"{MODEL_NAME}:{MODEL_VERSION}:{run_id}",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "status": "completed",
        "created_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_file": Path(source_path).name,
        "source_sha256": _sha256_file(source_path),
        "training_rows": int(features.shape[0]),
        "feature_names": FEATURE_NAMES,
        "parameters": MODEL_PARAMETERS,
        "metrics": {
            "physics_veto_count": int(physics_veto_count),
            "isolation_forest_outlier_count": int(outlier_mask.sum()),
            "isolation_forest_outlier_rate": float(outlier_mask.mean()),
            "decision_score_min": float(scores.min()),
            "decision_score_median": float(np.median(scores)),
            "decision_score_max": float(scores.max()),
        },
        "timing_seconds": {
            "fit": fit_seconds,
            "predict_and_score": predict_seconds,
            "total_model": fit_seconds + predict_seconds,
        },
        "environment": {
            "python_version": ".".join(map(str, __import__("sys").version_info[:3])),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _sha256_file(artifact_path),
        "artifact_validation": "passed_prediction_parity",
    }
    _write_json_atomic(run_artifact_dir / "metadata.json", metadata)
    _record_run(tracking_database, metadata)
    return outlier_mask, scores, metadata
