"""
Anomaly Detection Dashboard - Flask Backend
=============================================
Architecture: Task-Queue Pattern
  Browser  →  POST /api/run_lstm  →  spawns background thread  →  returns {task_id}
  Browser  →  polls GET /api/task/{id} every 2 s  →  {status, progress, results}
  Thread   →  trains LSTM  →  writes results to TASKS store  →  marks DONE

This mirrors how Celery+Redis works in production ML systems, implemented
using Python's threading module so no extra infrastructure is needed.
"""

import os
import json
import uuid
import io
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_store.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Task store  –  the in-memory job registry
# ---------------------------------------------------------------------------
# Each task looks like:
#   TASKS[task_id] = {
#     "status":   "pending" | "training" | "evaluating" | "done" | "error",
#     "progress": 0-100,          # epoch % during training
#     "message":  str,            # human-readable status line
#     "result":   dict | None,    # full result payload when done
#     "error":    str | None,
#     "created":  ISO timestamp
#   }
TASKS: dict = {}
TASKS_LOCK = threading.Lock()      # guards concurrent reads/writes

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
    Parse a '_cleaned.csv' file with columns:
    timestamp, scan_number, ch_00 [, ch_01 …], system_file, system_id
    """
    df = pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace")
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df, ch_cols


def parse_raw_instrument_csv(filepath):
    """
    Parse a raw DAQ instrument CSV (Cataluminescence, Biomass, Swiss, Default formats).
    Header detection uses the same keywords as the individual LSTM notebooks:
      'Scan Num', '101 (', 'Scan Swee', 'timestamp'
    Sentinel values (-9.9E+37) and hardware overloads (>10 000) are scrubbed to NaN.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        raise ValueError(f"Cannot read file: {e}")

    # ── Step 1: find the header row ──
    # Matches the notebook pattern: any keyword in the header line
    HEADER_KEYWORDS = ["Scan Num", "101 (", "Scan Swee", "timestamp", "Scan Number", "Scan Sweep"]
    data_start = None
    for i, line in enumerate(lines):
        if any(kw in line for kw in HEADER_KEYWORDS):
            data_start = i
            break

    if data_start is None:
        # Last resort: first non-blank line is the header
        for i, line in enumerate(lines):
            if line.strip():
                data_start = i
                break

    if data_start is None:
        raise ValueError("Could not detect a data header row in this CSV.")

    # ── Step 2: build DataFrame from header + data rows ──
    csv_text = "".join(lines[data_start:])
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise ValueError(f"CSV parse error: {e}")

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # ── Step 3: rename first column → timestamp, second → scan_number ──
    cols = list(df.columns)
    rename_map = {}
    if len(cols) >= 1:
        rename_map[cols[0]] = "timestamp"
    if len(cols) >= 2:
        rename_map[cols[1]] = "scan_number"
    df.rename(columns=rename_map, inplace=True)

    # ── Step 4: identify sensor/channel columns ──
    non_sensor = {"timestamp", "scan_number"}
    ch_cols = [c for c in df.columns if c not in non_sensor]

    # ── Step 5: numeric conversion + scrub sentinel values (notebook pattern) ──
    for col in ch_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Hardware overload sentinel: -9.9E+37 or any value > 10 000 in absolute
        overload = (df[col].abs() > 10_000) | (df[col] == np.inf) | (df[col] == -np.inf)
        df.loc[overload, col] = np.nan

    # ── Step 6: parse timestamp ──
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df, ch_cols


def load_file(filepath):
    """Auto-detect CSV format and parse — supports cleaned exports and raw instrument files."""
    fname = os.path.basename(filepath).lower()

    # Explicit cleaned-file marker
    if "_cleaned" in fname:
        return parse_cleaned_csv(filepath)

    # Peek at the first line to decide
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
    except Exception:
        first_line = ""

    if first_line.strip().lower().startswith("timestamp,scan_number"):
        return parse_cleaned_csv(filepath)

    # All instrument-format files go through the robust parser
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


# ---------------------------------------------------------------------------
# LSTM Autoencoder (PyTorch) – train-on-the-fly
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LSTMAutoencoder – exact architecture from the dataset notebooks
# (Biomass / Cataluminescence / Default: hidden_size=32  |  Swiss: hidden_size=64)
# hidden_size is auto-selected based on channel count, matching each notebook.
# ---------------------------------------------------------------------------

class LSTMAutoencoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Exact architecture used in all four dataset-specific LSTM notebooks.
    Encoder compresses the sequence to a bottleneck; decoder reconstructs it.
    Anomalies = sequences the model cannot reconstruct well (high MAE).
    """
    def __init__(self, num_features, hidden_size=32):
        if not TORCH_AVAILABLE:
            return
        super(LSTMAutoencoder, self).__init__()
        self.hidden_size = hidden_size

        # Encoder: reads the input sequence and produces a latent vector
        self.encoder_lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True
        )
        # Decoder: reconstructs the sequence from the latent vector
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        # Maps decoder output back to original feature space
        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Compress: encode the full sequence → bottleneck hidden state
        _, (hidden_state, _) = self.encoder_lstm(x)
        last_hidden_state = hidden_state[-1]          # shape: (batch, hidden)

        # Expand: repeat latent vector across every timestep for decoding
        repeated_hidden = last_hidden_state.unsqueeze(1).repeat(1, seq_len, 1)

        # Reconstruct
        decoded, _ = self.decoder_lstm(repeated_hidden)
        reconstructed = self.output_layer(decoded)
        return reconstructed


def _run_lstm(series_df, sensor_cols, time_steps=10, epochs=35, progress_cb=None):
    """
    Exact training + scoring pipeline from the dataset notebooks.
    progress_cb(epoch, total_epochs, avg_loss) is called after each epoch
    so the background thread can push live progress to the TASKS store.

    Steps (match every notebook exactly):
      1. StandardScaler normalisation
      2. Sliding time windows (time_steps)
      3. LSTMAutoencoder  –  hidden_size: 64 for ≥10 ch (Swiss), 32 otherwise
      4. L1Loss + Adam, batch_size=16, 35 epochs
      5. MAE reconstruction error per sequence
      6. Threshold = 97th percentile
      7. Confidence score 50-100 % for anomalies  (Complete_pipeline formula)
    """
    # 1. Normalise
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(series_df[sensor_cols]),
        columns=sensor_cols
    )

    # 2. Build sliding time windows
    Xs, ts_indices = [], []
    for i in range(len(scaled) - time_steps):
        Xs.append(scaled.iloc[i:i + time_steps].values)
        ts_indices.append(i + time_steps - 1)

    if len(Xs) < 10:
        raise ValueError(
            f"Not enough data for LSTM (got {len(Xs)} sequences, need >10). "
            "Try a smaller window or load a longer file."
        )

    X_np = np.array(Xs, dtype=np.float32)
    X_t  = torch.tensor(X_np)

    # 3. Auto-select hidden_size (matches each dataset notebook)
    num_features = len(sensor_cols)
    hidden_size  = 64 if num_features >= 10 else 32

    # 4. DataLoader – batch_size=16 matches every notebook
    loader = DataLoader(TensorDataset(X_t), batch_size=16, shuffle=False)

    # 5. Initialise model
    model     = LSTMAutoencoder(num_features=num_features, hidden_size=hidden_size)
    criterion = nn.L1Loss()          # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training loop with per-epoch progress reporting
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if progress_cb:
            progress_cb(epoch + 1, epochs, avg_loss)

    # 7. Evaluate
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_t)

    X_np_back = X_t.numpy()
    recon_np  = reconstructed.numpy()
    mae_loss  = np.mean(np.abs(recon_np - X_np_back), axis=(1, 2))

    # Threshold = 97th percentile  (identical to notebooks)
    threshold = np.percentile(mae_loss, 97)
    max_mae   = mae_loss.max()

    def _confidence(v):
        if v <= threshold or max_mae == threshold:
            return 0.0
        return min(100.0, 50.0 + (v - threshold) / (max_mae - threshold) * 50.0)

    flags  = (mae_loss > threshold).tolist()
    scores = mae_loss.tolist()
    confs  = [_confidence(v) for v in mae_loss]

    return flags, scores, confs, ts_indices, float(threshold)


def _lstm_task_worker(task_id, filepath, ch_cols, df, time_steps, epochs):
    """
    Background thread function.
    Writes progress updates to TASKS[task_id] as training proceeds,
    then writes the final result payload when done.
    This is exactly what a Celery worker does — just without Redis.
    """
    def _update(status, progress, message, result=None, error=None):
        with TASKS_LOCK:
            TASKS[task_id].update({
                "status":   status,
                "progress": progress,
                "message":  message,
                "result":   result,
                "error":    error,
            })

    def _progress_cb(epoch, total, avg_loss):
        pct = int(epoch / total * 90)   # reserve 10 % for evaluate step
        _update("training", pct,
                f"Epoch {epoch}/{total}  —  avg MAE loss: {avg_loss:.5f}")

    try:
        _update("training", 0, "Preparing data and building model…")

        flags, scores, confs, seq_indices, threshold = _run_lstm(
            df, ch_cols,
            time_steps=time_steps,
            epochs=epochs,
            progress_cb=_progress_cb,
        )

        _update("evaluating", 91, "Computing reconstruction errors and confidence scores…")

        timestamps = (
            df["timestamp"].astype(str).tolist()
            if "timestamp" in df.columns
            else list(range(len(df)))
        )

        anomaly_indices = [seq_indices[i] for i, f in enumerate(flags) if f]
        anomaly_scores  = [scores[i]      for i, f in enumerate(flags) if f]
        anomaly_confs   = [confs[i]       for i, f in enumerate(flags) if f]
        anomaly_ts      = [timestamps[idx] for idx in anomaly_indices]

        per_channel_values = {}
        for col in ch_cols:
            vals = df[col].tolist()
            per_channel_values[col] = [
                (None if (isinstance(vals[idx], float) and np.isnan(vals[idx])) else vals[idx])
                for idx in anomaly_indices
            ]

        all_mae = [
            {
                "index":      seq_indices[i],
                "timestamp":  timestamps[seq_indices[i]],
                "mae":        scores[i],
                "confidence": confs[i],
            }
            for i in range(len(scores))
        ]

        result_payload = {
            "algorithm":          "lstm_autoencoder",
            "channels":           ch_cols,
            "total_points":       len(df),
            "time_steps":         time_steps,
            "epochs":             epochs,
            "hidden_size":        64 if len(ch_cols) >= 10 else 32,
            "threshold":          threshold,
            "anomaly_count":      len(anomaly_indices),
            "anomaly_indices":    anomaly_indices,
            "anomaly_scores":     anomaly_scores,
            "anomaly_confidences": anomaly_confs,
            "anomaly_timestamps": anomaly_ts,
            "per_channel_values": per_channel_values,
            "all_mae":            all_mae,
            # Summary metrics (mirrors the notebook's printed report)
            "mean_mae":   float(np.mean([s for s in scores])),
            "max_mae":    float(np.max([s for s in scores])),
            "normal_count": len(df) - len(anomaly_indices),
        }

        _update("done", 100, f"Done — {len(anomaly_indices)} anomalies detected.", result=result_payload)

    except Exception as exc:
        _update("error", 0, str(exc), error=str(exc))


# ---------------------------------------------------------------------------
# Routes – async LSTM
# ---------------------------------------------------------------------------

@app.route("/api/run_lstm", methods=["POST"])
def run_lstm_async():
    """
    Spawn a background thread to train the LSTM and return {task_id} immediately.
    The frontend then polls /api/task/<task_id> for live progress.
    """
    if not TORCH_AVAILABLE:
        return jsonify({"error": "PyTorch not installed on server"}), 500

    body       = request.json
    experiment = body.get("experiment", "")
    filename   = body.get("filename", "")
    time_steps = int(body.get("time_steps", 10))
    epochs     = int(body.get("epochs", 35))

    filepath = os.path.join(UPLOAD_FOLDER, experiment, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        df, ch_cols = load_file(filepath)
    except Exception as e:
        return jsonify({"error": f"Parse error: {str(e)}"}), 400

    if not ch_cols:
        return jsonify({"error": "No sensor channels found"}), 400

    # Impute NaN before handing off to the thread  (same as notebooks)
    df[ch_cols] = df[ch_cols].ffill().bfill()

    # Register the task
    task_id = str(uuid.uuid4())
    with TASKS_LOCK:
        TASKS[task_id] = {
            "status":   "pending",
            "progress": 0,
            "message":  "Queued — waiting to start…",
            "result":   None,
            "error":    None,
            "created":  datetime.now().isoformat(),
            "meta": {
                "experiment": experiment,
                "filename":   filename,
                "channels":   ch_cols,
                "epochs":     epochs,
                "time_steps": time_steps,
            }
        }

    # Spawn background thread — does NOT block the HTTP response
    t = threading.Thread(
        target=_lstm_task_worker,
        args=(task_id, filepath, ch_cols, df, time_steps, epochs),
        daemon=True,
    )
    t.start()

    return jsonify({"task_id": task_id, "status": "pending"})


@app.route("/api/task/<task_id>", methods=["GET"])
def get_task(task_id):
    """
    Poll endpoint — the frontend calls this every 2 s to get live progress.
    Returns the full result payload once status == 'done'.
    """
    with TASKS_LOCK:
        task = TASKS.get(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@app.route("/api/tasks", methods=["GET"])
def list_tasks():
    """Return all tasks (for debugging/history view)."""
    with TASKS_LOCK:
        snapshot = {tid: {k: v for k, v in t.items() if k != "result"}
                    for tid, t in TASKS.items()}
    return jsonify(snapshot)





@app.route("/api/confidence_interval", methods=["POST"])
def confidence_interval():
    """Return rolling mean, upper CI, and lower CI bands for a channel."""
    body = request.json
    experiment = body.get("experiment", "")
    filename = body.get("filename", "")
    channel = body.get("channel", "")
    window = int(body.get("window", 20))
    n_sigma = float(body.get("n_sigma", 2.0))

    filepath = os.path.join(UPLOAD_FOLDER, experiment, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    df, ch_cols = load_file(filepath)
    if channel not in ch_cols:
        return jsonify({"error": "Channel not found"}), 400

    series = df[channel].copy()
    roll_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    roll_std  = series.rolling(window=window, center=True, min_periods=1).std().fillna(0)

    upper = (roll_mean + n_sigma * roll_std).tolist()
    lower = (roll_mean - n_sigma * roll_std).tolist()
    mean  = roll_mean.tolist()

    def clean(lst):
        return [None if (isinstance(v, float) and np.isnan(v)) else v for v in lst]

    return jsonify({
        "channel": channel,
        "window": window,
        "n_sigma": n_sigma,
        "mean": clean(mean),
        "upper": clean(upper),
        "lower": clean(lower),
    })


@app.route("/api/compare", methods=["POST"])
def compare_algorithms():
    """Run all 5 classical algorithms on a channel and return consensus results."""
    body = request.json
    experiment = body.get("experiment", "")
    filename = body.get("filename", "")
    channel = body.get("channel", "")

    filepath = os.path.join(UPLOAD_FOLDER, experiment, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        df, ch_cols = load_file(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if channel not in ch_cols:
        return jsonify({"error": "Channel not found"}), 400

    series = df[channel].copy()
    timestamps = df["timestamp"].astype(str).tolist() if "timestamp" in df.columns else list(range(len(df)))

    results = {}
    vote_matrix = np.zeros(len(series), dtype=int)

    algo_configs = {
        "zscore":           (detect_zscore,           {}),
        "iqr":              (detect_iqr,               {}),
        "isolation_forest": (detect_isolation_forest,  {}),
        "lof":              (detect_lof,               {}),
        "rolling_stats":    (detect_rolling_stats,     {}),
    }

    for name, (fn, kwargs) in algo_configs.items():
        try:
            mask, scores = fn(series, **kwargs)
            vote_matrix += mask.astype(int)
            results[name] = {
                "anomaly_count": int(mask.sum()),
                "anomaly_indices": np.where(mask)[0].tolist(),
            }
        except Exception:
            results[name] = {"anomaly_count": 0, "anomaly_indices": []}

    # Consensus: flagged by ≥ 2 algorithms
    consensus_mask = vote_matrix >= 2
    consensus_indices = np.where(consensus_mask)[0].tolist()
    vote_counts = vote_matrix.tolist()

    consensus_values = series.iloc[consensus_indices].tolist()
    consensus_ts     = [timestamps[i] for i in consensus_indices]

    return jsonify({
        "channel": channel,
        "total_points": len(series),
        "per_algorithm": results,
        "vote_counts": vote_counts,
        "consensus_indices": consensus_indices,
        "consensus_values": [None if (isinstance(v, float) and np.isnan(v)) else v for v in consensus_values],
        "consensus_timestamps": consensus_ts,
        "consensus_count": len(consensus_indices),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

