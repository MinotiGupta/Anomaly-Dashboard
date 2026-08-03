"""
Feedback Database — SQLite-backed persistent storage for human-in-the-loop labels.
==================================================================================

Schema:
  feedback_labels  — individual point-level labels (anomaly / normal)
  training_runs    — history of every LSTM training run and its config
  model_metrics    — per-run metrics (MAE, threshold, anomaly count)

Why SQLite instead of JSON?
  - Concurrent reads/writes are safe (WAL mode)
  - Queryable: "show all confirmed anomalies for this file"
  - Scalable: handles 100K+ labels without loading everything into memory
  - No extra infrastructure (unlike Redis/Postgres)
"""

import os
import json
import sqlite3
import threading
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.db")

# One connection per thread — SQLite requirement
_local = threading.local()


def _get_conn():
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback_labels (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_key    TEXT NOT NULL,               -- "experiment/filename.csv"
            channel     TEXT DEFAULT '',              -- channel name (or '' for multi-channel LSTM)
            point_index INTEGER NOT NULL,             -- row index in the CSV
            timestamp   TEXT,                         -- the data point's timestamp
            value       REAL,                         -- the sensor value at that point
            mae_score   REAL,                         -- reconstruction error from LSTM
            confidence  REAL,                         -- model confidence %
            label       TEXT NOT NULL DEFAULT '',     -- "anomaly" | "normal" | ""
            note        TEXT DEFAULT '',              -- optional user note
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            UNIQUE(file_key, channel, point_index)
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT UNIQUE NOT NULL,         -- UUID from the task queue
            file_key    TEXT NOT NULL,
            epochs      INTEGER,
            time_steps  INTEGER,
            hidden_size INTEGER,
            num_channels INTEGER,
            total_points INTEGER,
            anomaly_count INTEGER,
            threshold   REAL,
            mean_mae    REAL,
            max_mae     REAL,
            feedback_incorporated INTEGER DEFAULT 0,  -- how many feedback labels were used
            created_at  TEXT NOT NULL,
            status      TEXT DEFAULT 'completed'
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_file  ON feedback_labels(file_key);
        CREATE INDEX IF NOT EXISTS idx_feedback_label  ON feedback_labels(label);
        CREATE INDEX IF NOT EXISTS idx_runs_file       ON training_runs(file_key);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Feedback CRUD operations
# ---------------------------------------------------------------------------

def save_label(file_key, channel, point_index, label, note="",
               timestamp=None, value=None, mae_score=None, confidence=None):
    """
    Insert or update a single feedback label.
    Uses UPSERT (INSERT OR REPLACE) so re-labelling a point overwrites cleanly.
    """
    conn = _get_conn()
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT INTO feedback_labels
            (file_key, channel, point_index, timestamp, value, mae_score, confidence,
             label, note, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_key, channel, point_index) DO UPDATE SET
            label      = excluded.label,
            note       = excluded.note,
            mae_score  = excluded.mae_score,
            confidence = excluded.confidence,
            updated_at = excluded.updated_at
    """, (file_key, channel, point_index, timestamp, value,
          mae_score, confidence, label, note, now, now))
    conn.commit()


def save_labels_bulk(file_key, channel, labels_list):
    """
    Bulk insert/update feedback labels.
    labels_list: list of dicts with keys:
      point_index, label, note, timestamp, value, mae_score, confidence
    """
    conn = _get_conn()
    now = datetime.now().isoformat()
    for item in labels_list:
        conn.execute("""
            INSERT INTO feedback_labels
                (file_key, channel, point_index, timestamp, value, mae_score, confidence,
                 label, note, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_key, channel, point_index) DO UPDATE SET
                label      = excluded.label,
                note       = excluded.note,
                mae_score  = excluded.mae_score,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
        """, (
            file_key, channel,
            item.get("point_index"),
            item.get("timestamp"),
            item.get("value"),
            item.get("mae_score"),
            item.get("confidence"),
            item.get("label", ""),
            item.get("note", ""),
            now, now
        ))
    conn.commit()


def get_labels(file_key, channel=""):
    """Get all feedback labels for a file (optionally filtered by channel)."""
    conn = _get_conn()
    if channel:
        rows = conn.execute(
            "SELECT * FROM feedback_labels WHERE file_key=? AND channel=? ORDER BY point_index",
            (file_key, channel)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM feedback_labels WHERE file_key=? ORDER BY point_index",
            (file_key,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_confirmed_anomalies(file_key, channel=""):
    """Get indices of points the user confirmed as real anomalies."""
    conn = _get_conn()
    if channel:
        rows = conn.execute(
            "SELECT point_index FROM feedback_labels WHERE file_key=? AND channel=? AND label='anomaly'",
            (file_key, channel)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT point_index FROM feedback_labels WHERE file_key=? AND label='anomaly'",
            (file_key,)
        ).fetchall()
    return [r["point_index"] for r in rows]


def get_confirmed_normals(file_key, channel=""):
    """Get indices of points the user confirmed as normal (false positives)."""
    conn = _get_conn()
    if channel:
        rows = conn.execute(
            "SELECT point_index FROM feedback_labels WHERE file_key=? AND channel=? AND label='normal'",
            (file_key, channel)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT point_index FROM feedback_labels WHERE file_key=? AND label='normal'",
            (file_key,)
        ).fetchall()
    return [r["point_index"] for r in rows]


def get_feedback_summary(file_key):
    """Return counts of anomaly/normal/unlabelled for a file."""
    conn = _get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*)                                      AS total,
            SUM(CASE WHEN label='anomaly' THEN 1 ELSE 0 END) AS confirmed_anomalies,
            SUM(CASE WHEN label='normal'  THEN 1 ELSE 0 END) AS confirmed_normals,
            SUM(CASE WHEN label=''        THEN 1 ELSE 0 END) AS unlabelled
        FROM feedback_labels WHERE file_key=?
    """, (file_key,)).fetchone()
    return dict(row) if row else {"total": 0, "confirmed_anomalies": 0, "confirmed_normals": 0, "unlabelled": 0}


def delete_label(file_key, channel, point_index):
    """Remove a specific feedback label."""
    conn = _get_conn()
    conn.execute(
        "DELETE FROM feedback_labels WHERE file_key=? AND channel=? AND point_index=?",
        (file_key, channel, point_index)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Training run history
# ---------------------------------------------------------------------------

def record_training_run(task_id, file_key, epochs, time_steps, hidden_size,
                        num_channels, total_points, anomaly_count, threshold,
                        mean_mae, max_mae, feedback_incorporated=0):
    """Record a completed training run in the database."""
    conn = _get_conn()
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO training_runs
            (task_id, file_key, epochs, time_steps, hidden_size, num_channels,
             total_points, anomaly_count, threshold, mean_mae, max_mae,
             feedback_incorporated, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (task_id, file_key, epochs, time_steps, hidden_size, num_channels,
          total_points, anomaly_count, threshold, mean_mae, max_mae,
          feedback_incorporated, now))
    conn.commit()


def get_training_history(file_key=None, limit=20):
    """Get recent training runs, optionally filtered by file."""
    conn = _get_conn()
    if file_key:
        rows = conn.execute(
            "SELECT * FROM training_runs WHERE file_key=? ORDER BY created_at DESC LIMIT ?",
            (file_key, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# Initialise DB on import
init_db()
