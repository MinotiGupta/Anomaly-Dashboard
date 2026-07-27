"""
Anomaly Detection Dashboard - Flask Backend
=============================================
Provides APIs for:
  - Listing & loading experiment data from uploads/
  - Uploading new CSV files
  - Running unsupervised anomaly detection (Z-Score, IQR, Isolation Forest, LOF, Rolling Stats)
  - Human-in-the-loop feedback for anomaly labels
  - Exporting annotated data
"""

import os
import json
import uuid
import io
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_store.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Feedback persistence
# ---------------------------------------------------------------------------

def load_feedback():
    """Load human feedback from disk."""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {}


def save_feedback(data):
    """Persist human feedback to disk."""
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Data parsing helpers
# ---------------------------------------------------------------------------

def parse_cleaned_csv(filepath):
    """
    Parse a '_cleaned.csv' file that has columns:
    timestamp, scan_number, ch_00 [, ch_01 …], system_file, system_id
    """
    df = pd.read_csv(filepath)
    # Identify channel columns
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df, ch_cols


def parse_raw_instrument_csv(filepath):
    """
    Parse a raw DAQ instrument CSV export.
    The data section starts after a row whose first cell contains 'Scan Sweep Time'.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if "Scan Sweep Time" in line or "Scan Number" in line:
            data_start = i
            break

    if data_start is None:
        # Fallback: try to read as plain CSV
        df = pd.read_csv(filepath)
        ch_cols = [c for c in df.columns if c not in ("timestamp", "scan_number", "system_file", "system_id")]
        return df, ch_cols

    # Read header row and data rows
    header_line = lines[data_start]
    data_lines = lines[data_start + 1:]

    # Build a CSV string
    csv_text = header_line + "".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text))

    # Clean up column names
    df.columns = [c.strip() for c in df.columns]

    # Rename first two columns
    cols = list(df.columns)
    rename_map = {}
    if len(cols) >= 1:
        rename_map[cols[0]] = "timestamp"
    if len(cols) >= 2:
        rename_map[cols[1]] = "scan_number"
    df.rename(columns=rename_map, inplace=True)

    # Drop entirely-empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Identify channel columns (everything except timestamp and scan_number)
    ch_cols = [c for c in df.columns if c not in ("timestamp", "scan_number")]

    # Replace sentinel values (-9.9E+37) with NaN
    df.replace(-9.9e+37, np.nan, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df, ch_cols


def load_file(filepath):
    """Auto-detect format and parse."""
    fname = os.path.basename(filepath).lower()
    if "_cleaned" in fname:
        return parse_cleaned_csv(filepath)
    else:
        # Try to detect if it's a raw instrument file
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        if first_line.strip().startswith("timestamp,scan_number"):
            return parse_cleaned_csv(filepath)
        else:
            return parse_raw_instrument_csv(filepath)


# ---------------------------------------------------------------------------
# Anomaly detection algorithms (all unsupervised)
# ---------------------------------------------------------------------------

def detect_zscore(series, threshold=3.0):
    """Z-Score based anomaly detection."""
    clean = series.dropna()
    if len(clean) < 3:
        return np.zeros(len(series), dtype=bool), np.zeros(len(series))
    z = np.abs(scipy_stats.zscore(clean))
    mask = np.zeros(len(series), dtype=bool)
    scores = np.zeros(len(series))
    mask[clean.index] = z > threshold
    scores[clean.index] = z
    return mask, scores


def detect_iqr(series, factor=1.5):
    """IQR-based outlier detection."""
    clean = series.dropna()
    if len(clean) < 4:
        return np.zeros(len(series), dtype=bool), np.zeros(len(series))
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = np.zeros(len(series), dtype=bool)
    scores = np.zeros(len(series))
    mask[clean.index] = (clean < lower) | (clean > upper)
    # Score = distance from nearest fence, normalised by IQR
    dist = np.where(clean < lower, lower - clean, np.where(clean > upper, clean - upper, 0))
    scores[clean.index] = dist / (iqr + 1e-10)
    return mask, scores


def detect_isolation_forest(series, contamination=0.05):
    """Isolation Forest anomaly detection."""
    clean = series.dropna()
    if len(clean) < 10:
        return np.zeros(len(series), dtype=bool), np.zeros(len(series))
    X = clean.values.reshape(-1, 1)
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = clf.fit_predict(X)
    raw_scores = -clf.score_samples(X)  # higher = more anomalous
    mask = np.zeros(len(series), dtype=bool)
    scores = np.zeros(len(series))
    mask[clean.index] = preds == -1
    scores[clean.index] = raw_scores
    return mask, scores


def detect_lof(series, n_neighbors=20, contamination=0.05):
    """Local Outlier Factor anomaly detection."""
    clean = series.dropna()
    if len(clean) < max(10, n_neighbors + 1):
        return np.zeros(len(series), dtype=bool), np.zeros(len(series))
    X = clean.values.reshape(-1, 1)
    n = min(n_neighbors, len(X) - 1)
    clf = LocalOutlierFactor(n_neighbors=n, contamination=contamination)
    preds = clf.fit_predict(X)
    raw_scores = -clf.negative_outlier_factor_  # higher = more anomalous
    mask = np.zeros(len(series), dtype=bool)
    scores = np.zeros(len(series))
    mask[clean.index] = preds == -1
    scores[clean.index] = raw_scores
    return mask, scores


def detect_rolling_stats(series, window=20, sigma=3.0):
    """Rolling mean/std anomaly detection – good for detecting sudden amplitude changes."""
    clean = series.dropna()
    if len(clean) < window + 1:
        return np.zeros(len(series), dtype=bool), np.zeros(len(series))
    rolling_mean = clean.rolling(window=window, center=True).mean()
    rolling_std = clean.rolling(window=window, center=True).std()
    z = np.abs((clean - rolling_mean) / (rolling_std + 1e-10))
    mask = np.zeros(len(series), dtype=bool)
    scores = np.zeros(len(series))
    z_filled = z.fillna(0)
    mask[clean.index] = z_filled > sigma
    scores[clean.index] = z_filled.values
    return mask, scores


ALGO_MAP = {
    "zscore": detect_zscore,
    "iqr": detect_iqr,
    "isolation_forest": detect_isolation_forest,
    "lof": detect_lof,
    "rolling_stats": detect_rolling_stats,
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/experiments")
def list_experiments():
    """List all experiment folders and their CSV files."""
    experiments = []
    for folder in sorted(os.listdir(UPLOAD_FOLDER)):
        folder_path = os.path.join(UPLOAD_FOLDER, folder)
        if os.path.isdir(folder_path):
            files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".csv") and f != ".keep"
            ]
            if files:
                experiments.append({
                    "name": folder,
                    "files": sorted(files),
                })
    return jsonify(experiments)


@app.route("/api/load", methods=["POST"])
def load_data():
    """Load a specific CSV file and return parsed data."""
    body = request.json
    experiment = body.get("experiment", "")
    filename = body.get("filename", "")
    filepath = os.path.join(UPLOAD_FOLDER, experiment, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        df, ch_cols = load_file(filepath)
    except Exception as e:
        return jsonify({"error": f"Parse error: {str(e)}"}), 400

    # Build response
    timestamps = df["timestamp"].astype(str).tolist() if "timestamp" in df.columns else list(range(len(df)))
    channels = {}
    for col in ch_cols:
        vals = df[col].tolist()
        # Replace NaN with None for JSON
        channels[col] = [None if (isinstance(v, float) and np.isnan(v)) else v for v in vals]

    return jsonify({
        "timestamps": timestamps,
        "channels": channels,
        "channel_names": ch_cols,
        "num_points": len(df),
        "file_key": f"{experiment}/{filename}",
    })


@app.route("/api/detect", methods=["POST"])
def detect_anomalies():
    """Run anomaly detection on a loaded dataset."""
    body = request.json
    experiment = body.get("experiment", "")
    filename = body.get("filename", "")
    channel = body.get("channel", "")
    algorithm = body.get("algorithm", "zscore")
    params = body.get("params", {})

    filepath = os.path.join(UPLOAD_FOLDER, experiment, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        df, ch_cols = load_file(filepath)
    except Exception as e:
        return jsonify({"error": f"Parse error: {str(e)}"}), 400

    if channel not in ch_cols:
        return jsonify({"error": f"Channel '{channel}' not found"}), 400

    series = df[channel].copy()

    algo_fn = ALGO_MAP.get(algorithm)
    if algo_fn is None:
        return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

    # Pass algorithm-specific params
    try:
        if algorithm == "zscore":
            threshold = float(params.get("threshold", 3.0))
            mask, scores = algo_fn(series, threshold=threshold)
        elif algorithm == "iqr":
            factor = float(params.get("factor", 1.5))
            mask, scores = algo_fn(series, factor=factor)
        elif algorithm == "isolation_forest":
            contamination = float(params.get("contamination", 0.05))
            mask, scores = algo_fn(series, contamination=contamination)
        elif algorithm == "lof":
            n_neighbors = int(params.get("n_neighbors", 20))
            contamination = float(params.get("contamination", 0.05))
            mask, scores = algo_fn(series, n_neighbors=n_neighbors, contamination=contamination)
        elif algorithm == "rolling_stats":
            window = int(params.get("window", 20))
            sigma = float(params.get("sigma", 3.0))
            mask, scores = algo_fn(series, window=window, sigma=sigma)
        else:
            mask, scores = algo_fn(series)
    except Exception as e:
        return jsonify({"error": f"Detection error: {str(e)}"}), 500

    anomaly_indices = np.where(mask)[0].tolist()
    anomaly_values = series.iloc[anomaly_indices].tolist() if anomaly_indices else []
    anomaly_scores = scores[anomaly_indices].tolist() if anomaly_indices else []

    timestamps = df["timestamp"].astype(str).tolist() if "timestamp" in df.columns else list(range(len(df)))
    anomaly_timestamps = [timestamps[i] for i in anomaly_indices]

    return jsonify({
        "algorithm": algorithm,
        "channel": channel,
        "total_points": len(series),
        "anomaly_count": len(anomaly_indices),
        "anomaly_indices": anomaly_indices,
        "anomaly_values": [None if (isinstance(v, float) and np.isnan(v)) else v for v in anomaly_values],
        "anomaly_scores": anomaly_scores,
        "anomaly_timestamps": anomaly_timestamps,
    })


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Submit human-in-the-loop feedback for anomaly points."""
    body = request.json
    file_key = body.get("file_key", "")
    channel = body.get("channel", "")
    feedback_points = body.get("feedback_points", [])
    # Each point: { index, value, timestamp, label: "anomaly"|"normal", note: "" }

    all_feedback = load_feedback()
    key = f"{file_key}::{channel}"
    if key not in all_feedback:
        all_feedback[key] = []

    for pt in feedback_points:
        pt["submitted_at"] = datetime.now().isoformat()
        # Update existing or append
        existing = [p for p in all_feedback[key] if p.get("index") == pt.get("index")]
        if existing:
            existing[0].update(pt)
        else:
            all_feedback[key].append(pt)

    save_feedback(all_feedback)
    return jsonify({"status": "ok", "total_feedback": len(all_feedback[key])})


@app.route("/api/feedback/get", methods=["POST"])
def get_feedback():
    """Retrieve stored feedback for a file/channel."""
    body = request.json
    file_key = body.get("file_key", "")
    channel = body.get("channel", "")
    key = f"{file_key}::{channel}"
    all_feedback = load_feedback()
    return jsonify(all_feedback.get(key, []))


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload a new CSV file to an experiment folder."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    experiment = request.form.get("experiment", "uploaded")

    folder = os.path.join(UPLOAD_FOLDER, experiment)
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, file.filename)
    file.save(filepath)

    return jsonify({"status": "ok", "path": f"{experiment}/{file.filename}"})


@app.route("/api/export", methods=["POST"])
def export_annotated():
    """Export data with anomaly labels and human feedback as CSV."""
    body = request.json
    experiment = body.get("experiment", "")
    filename = body.get("filename", "")
    channel = body.get("channel", "")
    anomaly_indices = body.get("anomaly_indices", [])

    filepath = os.path.join(UPLOAD_FOLDER, experiment, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    df, ch_cols = load_file(filepath)

    # Add anomaly label column
    df["anomaly_detected"] = False
    if anomaly_indices:
        df.loc[anomaly_indices, "anomaly_detected"] = True

    # Merge human feedback
    file_key = f"{experiment}/{filename}"
    key = f"{file_key}::{channel}"
    all_feedback = load_feedback()
    feedback_list = all_feedback.get(key, [])

    df["human_label"] = ""
    df["human_note"] = ""
    for fb in feedback_list:
        idx = fb.get("index")
        if idx is not None and 0 <= idx < len(df):
            df.at[idx, "human_label"] = fb.get("label", "")
            df.at[idx, "human_note"] = fb.get("note", "")

    # Convert to CSV in memory
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"annotated_{filename}",
    )


@app.route("/api/stats", methods=["POST"])
def get_stats():
    """Get summary statistics for a channel."""
    body = request.json
    experiment = body.get("experiment", "")
    filename = body.get("filename", "")
    channel = body.get("channel", "")

    filepath = os.path.join(UPLOAD_FOLDER, experiment, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    df, ch_cols = load_file(filepath)
    if channel not in ch_cols:
        return jsonify({"error": "Channel not found"}), 400

    series = df[channel].dropna()
    return jsonify({
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
