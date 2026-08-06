"""
Anomaly Detection Dashboard - Backend API
Runs the actual LSTM Autoencoder pipeline from Complete_pipeline.ipynb
Includes: feedback loop, data classification, metadata extraction, retraining
"""

import os
import uuid
import json
import math
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import traceback


# ====================================================================
# CUSTOM JSON PROVIDER - converts NaN / Inf to null
# ====================================================================
class SafeJSONProvider(DefaultJSONProvider):
    """Replaces NaN and Inf with None so JSON stays valid."""

    def dumps(self, obj, **kwargs):
        return json.dumps(obj, default=self.default, allow_nan=False, **kwargs)

    def default(self, o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def sanitize(obj):
    """Recursively walk a dict/list and replace NaN/Inf floats with None."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


app = Flask(__name__)
app.json_provider_class = SafeJSONProvider
app.json = SafeJSONProvider(app)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Store run results in memory (keyed by run_id)
results_store = {}

# Known data types (matching folder names in C:\dev\AnomalyDetection\data)
KNOWN_DATA_TYPES = ["biomass", "cataluminescence", "swiss_roll", "default"]


# ====================================================================
# SQLITE FEEDBACK DATABASE
# ====================================================================
def get_db_path(data_type):
    """Each data type gets its own SQLite database file."""
    safe_name = data_type.lower().replace(" ", "_")
    return os.path.join(FEEDBACK_DIR, f"{safe_name}_feedback.db")


def init_feedback_db(data_type):
    """Create the feedback table if it doesn't exist."""
    db_path = get_db_path(data_type)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            data_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            mae_score REAL NOT NULL,
            confidence REAL NOT NULL,
            sensor_values TEXT NOT NULL,
            is_true_anomaly INTEGER,
            feedback_time TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(data_type, run_id, file_name, timestamp, mae_score,
                  confidence, sensor_values, is_true_anomaly):
    """Save a single feedback entry."""
    init_feedback_db(data_type)
    db_path = get_db_path(data_type)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO feedback
           (run_id, file_name, data_type, timestamp, mae_score, confidence,
            sensor_values, is_true_anomaly)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, file_name, data_type, timestamp, mae_score, confidence,
         json.dumps(sensor_values), 1 if is_true_anomaly else 0)
    )
    conn.commit()
    conn.close()


def get_feedback_for_type(data_type):
    """Retrieve all feedback entries for a data type."""
    init_feedback_db(data_type)
    db_path = get_db_path(data_type)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM feedback ORDER BY feedback_time DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_confirmed_anomaly_timestamps(data_type):
    """Get timestamps confirmed as TRUE anomalies by the user."""
    init_feedback_db(data_type)
    db_path = get_db_path(data_type)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT timestamp FROM feedback WHERE is_true_anomaly = 1"
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def get_confirmed_normal_timestamps(data_type):
    """Get timestamps confirmed as NOT anomalies by the user (false positives)."""
    init_feedback_db(data_type)
    db_path = get_db_path(data_type)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT timestamp FROM feedback WHERE is_true_anomaly = 0"
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


# ====================================================================
# DATA TYPE CLASSIFICATION
# ====================================================================
def classify_data_type(file_path, df, sensor_cols):
    """
    Classify uploaded CSV into one of the known data types by inspecting
    column patterns and file header structure.
    """
    cols_lower = [c.lower() for c in df.columns]

    # Biomass data uses ch_00, ch_01, ... column naming
    if any("ch_0" in c for c in cols_lower):
        return "biomass"

    # Check first few lines of the raw file for instrument model hints
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header_lines = [f.readline() for _ in range(6)]
        header_text = " ".join(header_lines).lower()
    except Exception:
        header_text = ""

    # Cataluminescence uses DAQ970A model
    if "daq970a" in header_text:
        return "cataluminescence"

    # Default uses 34970A model with 5 temperature channels
    if "34970a" in header_text:
        temp_count = sum(1 for c in sensor_cols if "°C" in c or "°c" in c.lower())
        if temp_count <= 5:
            return "default"
        else:
            return "swiss_roll"

    # Fallback: if columns contain °C, classify by channel count
    temp_cols = [c for c in sensor_cols if "°C" in c]
    if len(temp_cols) > 5:
        return "swiss_roll"
    if len(temp_cols) > 0:
        return "default"

    return "unknown"


# ====================================================================
# INSTRUMENT METADATA EXTRACTION (from defaul_csv_analysis.ipynb)
# ====================================================================
def extract_instrument_metadata(file_path):
    """
    For files with Agilent/Keysight instrument headers (default, swiss_roll,
    cataluminescence), extract metadata fields like Model, Serial, Firmware,
    Start/Stop Time, etc.
    Returns None if no metadata is found.
    """
    metadata = {}
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 30:  # Only scan first 30 lines for metadata
                    break
                lines.append(line.strip())

        # Check if this file has instrument headers
        first_line = lines[0] if lines else ""
        if "Address" not in first_line and "Model" not in first_line:
            return None

        # Parse key-value metadata from the header rows
        metadata_keys = {
            "Address": "address",
            "Model": "model",
            "Serial Number": "serial_number",
            "Firmware Version": "firmware_version",
            "Start Time": "start_time",
            "Stop Time": "stop_time",
            "Total Channels": "total_channels",
        }

        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2:
                key = parts[0].strip().rstrip(":")
                value = parts[1].strip()
                if key in metadata_keys and value:
                    metadata[metadata_keys[key]] = value

        # Extract channel configuration
        channel_config = []
        in_channel_section = False
        for line in lines:
            if "Channel Configuration" in line:
                in_channel_section = True
                continue
            if in_channel_section and line.startswith("Channels,"):
                continue  # Skip header row
            if in_channel_section:
                parts = line.split(",")
                if parts[0].strip().isdigit() or (
                    len(parts[0].strip()) == 3 and parts[0].strip().startswith("1")
                ):
                    ch = {
                        "channel_id": parts[0].strip(),
                        "function": parts[2].strip() if len(parts) > 2 else "",
                        "range": parts[3].strip() if len(parts) > 3 else "",
                        "unit": parts[4].strip() if len(parts) > 4 else "",
                    }
                    channel_config.append(ch)

        if channel_config:
            metadata["channel_config"] = channel_config

        # Extract module info
        for line in lines:
            if line.startswith("Modules"):
                parts = line.split(",")
                module_info = [p.strip() for p in parts[1:] if p.strip()]
                if module_info:
                    metadata["modules"] = " / ".join(module_info)

        return metadata if metadata else None

    except Exception:
        return None


# ====================================================================
# MODEL ARCHITECTURE (from Complete_pipeline.ipynb)
# ====================================================================
class DynamicLSTMAutoencoder(nn.Module):
    def __init__(self, num_features):
        super(DynamicLSTMAutoencoder, self).__init__()
        self.hidden_size = min(64, max(16, num_features * 4))
        self.encoder_lstm = nn.LSTM(
            input_size=num_features, hidden_size=self.hidden_size, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_size, num_features)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (hidden_state, _) = self.encoder_lstm(x)
        last_hidden_state = hidden_state[-1]
        repeated_hidden = last_hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(repeated_hidden)
        reconstructed = self.output_layer(decoded)
        return reconstructed


# ====================================================================
# PIPELINE FUNCTIONS (from Complete_pipeline.ipynb)
# ====================================================================
def load_and_summarize_data(file_path):
    """Generalized data loader that auto-detects headers and sensor columns."""
    skip_rows = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if any(
                key in line
                for key in ["Scan Num", "101 (", "Scan Swee", "timestamp"]
            ):
                skip_rows = idx
                break

    df = pd.read_csv(file_path, skiprows=skip_rows)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

    # Identify time column
    time_cols = [
        col for col in df.columns if "time" in col.lower() or "swee" in col.lower()
    ]
    if time_cols:
        df["Timestamp"] = df[time_cols[0]]
    else:
        df["Timestamp"] = df.index

    # Identify sensor columns
    sensor_cols = [col for col in df.columns if "C" in col or "ch_" in col.lower()]
    # Exclude Timestamp if it crept in
    sensor_cols = [c for c in sensor_cols if c != "Timestamp"]

    # Scrub hardware overloads
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        overload_mask = (df[col].abs() > 10000) | (df[col] == np.inf)
        df.loc[overload_mask, col] = np.nan

    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    # Calculate average interval
    try:
        numeric_timestamps = pd.to_numeric(df["Timestamp"], errors="coerce")
        avg_interval = float(numeric_timestamps.diff().mean())
        if math.isnan(avg_interval) or math.isinf(avg_interval):
            avg_interval = None
    except Exception:
        avg_interval = None

    summary = {
        "total_rows": len(df),
        "total_channels": len(sensor_cols),
        "sensor_columns": sensor_cols,
        "avg_interval": avg_interval,
    }

    return df, sensor_cols, summary


def create_sequences(df_scaled, timestamps, time_steps):
    """Convert 2D tabular data into 3D tensors for LSTM processing."""
    Xs, ts = [], []
    for i in range(len(df_scaled) - time_steps):
        Xs.append(df_scaled.iloc[i : (i + time_steps)].values)
        ts.append(timestamps.iloc[i + time_steps - 1])
    return np.array(Xs), np.array(ts)


def train_autoencoder(model, train_loader, num_epochs=30, learning_rate=0.001):
    """Train the LSTM autoencoder and return per-epoch loss."""
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)

    return model, epoch_losses


def train_with_feedback(model, train_loader, X_tensor, timestamps,
                        confirmed_normal_ts, num_epochs=30, learning_rate=0.001):
    """
    Feedback-aware training: penalizes reconstruction of confirmed-normal
    timestamps less (they should reconstruct well) while keeping the
    standard autoencoder objective.
    """
    criterion = nn.L1Loss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses = []

    # Build weight mask: sequences corresponding to confirmed-normal timestamps
    # get higher weight (model should reconstruct them accurately)
    ts_list = list(timestamps)
    weights = torch.ones(len(ts_list), dtype=torch.float32)
    for i, ts in enumerate(ts_list):
        if str(ts) in confirmed_normal_ts:
            weights[i] = 2.0  # Upweight confirmed-normal so model learns them better

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        batch_start = 0
        for (batch_x,) in train_loader:
            batch_end = batch_start + len(batch_x)
            batch_weights = weights[batch_start:batch_end].unsqueeze(1).unsqueeze(2)
            batch_start = batch_end

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss_per_sample = criterion(outputs, batch_x)
            weighted_loss = (loss_per_sample * batch_weights).mean()
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += weighted_loss.item()
            batch_count += 1
        avg_loss = epoch_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)

    return model, epoch_losses


def detect_anomalies(
    model, X_seq_tensor, df_timestamps, df_original, sensor_cols, contamination=0.03
):
    """Detect anomalies and compute confidence scores."""
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_seq_tensor)

    X_seq_np = X_seq_tensor.numpy()
    reconstructed_np = reconstructed.numpy()

    # Per-sequence MAE
    mae_loss = np.mean(np.abs(reconstructed_np - X_seq_np), axis=(1, 2))

    # Per-feature MAE for the last timestep of each sequence
    per_feature_mae = np.mean(
        np.abs(reconstructed_np[:, -1, :] - X_seq_np[:, -1, :]), axis=0
    )

    threshold = np.percentile(mae_loss, (1 - contamination) * 100)
    max_mae = np.max(mae_loss)

    results_df = pd.DataFrame(
        {
            "Timestamp": df_timestamps,
            "MAE_Score": mae_loss,
            "Is_Anomaly": mae_loss > threshold,
        }
    )

    def calculate_confidence(mae):
        if mae <= threshold:
            return 0.0
        if max_mae == threshold:
            return 100.0
        scaled = 50.0 + ((mae - threshold) / (max_mae - threshold)) * 50.0
        return min(100.0, scaled)

    results_df["Confidence_Score"] = results_df["MAE_Score"].apply(
        calculate_confidence
    )

    merged_df = pd.merge(results_df, df_original, on="Timestamp", how="inner")
    anomalies_only = merged_df[merged_df["Is_Anomaly"] == True]

    return merged_df, anomalies_only, threshold, mae_loss, per_feature_mae


def run_pipeline(file_path, use_feedback=False):
    """Execute the full anomaly detection pipeline and return structured results."""
    file_name = os.path.basename(file_path)

    # Step 1: Load data
    df, sensors, summary = load_and_summarize_data(file_path)
    if len(sensors) == 0:
        raise ValueError("No valid sensor channels detected in the uploaded CSV.")

    # Step 1b: Classify data type
    data_type = classify_data_type(file_path, df, sensors)

    # Step 1c: Extract instrument metadata (if available)
    metadata = extract_instrument_metadata(file_path)

    # Step 2: Scale
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[sensors]), columns=sensors)

    # Step 3: Create sequences
    TIME_STEPS = 10
    X_seq, timestamps_seq = create_sequences(df_scaled, df["Timestamp"], TIME_STEPS)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Step 4: Build and train model
    model = DynamicLSTMAutoencoder(num_features=len(sensors))

    if use_feedback and data_type != "unknown":
        confirmed_normal = get_confirmed_normal_timestamps(data_type)
        if confirmed_normal:
            trained_model, epoch_losses = train_with_feedback(
                model, train_loader, X_tensor, timestamps_seq,
                confirmed_normal, num_epochs=30
            )
        else:
            trained_model, epoch_losses = train_autoencoder(
                model, train_loader, num_epochs=30
            )
    else:
        trained_model, epoch_losses = train_autoencoder(
            model, train_loader, num_epochs=30
        )

    # Step 5: Detect anomalies
    merged_df, anomalies_df, threshold, mae_scores, per_feature_mae = (
        detect_anomalies(trained_model, X_tensor, timestamps_seq, df, sensors)
    )

    # Build JSON-serializable results
    channels_data = {}
    for col in sensors:
        channels_data[col] = {
            "timestamps": merged_df["Timestamp"].astype(str).tolist(),
            "values": merged_df[col].tolist(),
            "anomaly_timestamps": anomalies_df["Timestamp"].astype(str).tolist(),
            "anomaly_values": anomalies_df[col].tolist(),
            "anomaly_confidences": anomalies_df["Confidence_Score"].tolist(),
        }

    # MAE distribution
    mae_hist, mae_bin_edges = np.histogram(mae_scores, bins=50)
    mae_distribution = {
        "counts": mae_hist.tolist(),
        "bin_edges": mae_bin_edges.tolist(),
        "threshold": float(threshold),
    }

    # Training loss curve
    training_loss = {
        "epochs": list(range(1, len(epoch_losses) + 1)),
        "losses": epoch_losses,
    }

    # Anomaly table
    anomaly_table = []
    for _, row in anomalies_df.iterrows():
        entry = {
            "timestamp": str(row["Timestamp"]),
            "confidence": round(row["Confidence_Score"], 1),
            "mae_score": round(row["MAE_Score"], 4),
            "sensor_values": {col: round(row[col], 4) for col in sensors},
        }
        anomaly_table.append(entry)

    # Per-feature contribution
    feature_contribution = {
        "features": sensors,
        "mae_values": per_feature_mae.tolist(),
    }

    # Per-channel statistics (mean, std, skewness, kurtosis, 95% CI)
    from scipy import stats as scipy_stats
    channel_statistics = {}
    for col in sensors:
        col_data = df[col].dropna()
        n = len(col_data)
        mean_val = float(col_data.mean())
        std_val = float(col_data.std())
        skew_val = float(col_data.skew())
        kurt_val = float(col_data.kurtosis())
        min_val = float(col_data.min())
        max_val = float(col_data.max())
        q1 = float(col_data.quantile(0.25))
        median_val = float(col_data.median())
        q3 = float(col_data.quantile(0.75))
        iqr = q3 - q1

        # 95% confidence interval for the mean
        if n > 1 and std_val > 0:
            se = std_val / np.sqrt(n)
            ci_low = mean_val - 1.96 * se
            ci_high = mean_val + 1.96 * se
        else:
            ci_low = mean_val
            ci_high = mean_val

        channel_statistics[col] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "q1": q1,
            "median": median_val,
            "q3": q3,
            "iqr": iqr,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "count": n,
        }

    # Confidence score distribution for anomalies
    anomaly_confidences = anomalies_df["Confidence_Score"].tolist()
    if anomaly_confidences:
        conf_hist, conf_bin_edges = np.histogram(anomaly_confidences, bins=20,
                                                  range=(50, 100))
        confidence_distribution = {
            "counts": conf_hist.tolist(),
            "bin_edges": conf_bin_edges.tolist(),
        }
    else:
        confidence_distribution = {
            "counts": [],
            "bin_edges": [],
        }

    # Load existing feedback for this data type
    existing_feedback = {}
    if data_type != "unknown":
        fb_entries = get_feedback_for_type(data_type)
        for fb in fb_entries:
            existing_feedback[fb["timestamp"]] = fb["is_true_anomaly"]

    return {
        "file_name": file_name,
        "data_type": data_type,
        "instrument_metadata": metadata,
        "summary": summary,
        "channels": channels_data,
        "mae_distribution": mae_distribution,
        "training_loss": training_loss,
        "anomaly_table": anomaly_table,
        "feature_contribution": feature_contribution,
        "channel_statistics": channel_statistics,
        "confidence_distribution": confidence_distribution,
        "total_anomalies": len(anomalies_df),
        "total_points": len(merged_df),
        "anomaly_rate": round(len(anomalies_df) / max(len(merged_df), 1) * 100, 2),
        "existing_feedback": existing_feedback,
        "used_feedback": use_feedback,
    }


# ====================================================================
# API ROUTES
# ====================================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """Upload a CSV file and return its ID."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    file.save(save_path)

    return jsonify(
        {"file_id": file_id, "filename": file.filename, "path": save_path}
    )


@app.route("/api/run", methods=["POST"])
def run_model():
    """Run the LSTM autoencoder pipeline on an uploaded CSV."""
    data = request.get_json()
    file_path = data.get("path")
    use_feedback = data.get("use_feedback", False)

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400

    try:
        results = sanitize(run_pipeline(file_path, use_feedback=use_feedback))
        run_id = str(uuid.uuid4())
        results_store[run_id] = results
        return jsonify({"run_id": run_id, "results": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Save user feedback on detected anomalies."""
    data = request.get_json()
    required = ["run_id", "data_type", "file_name", "feedback_entries"]
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing field: {key}"}), 400

    data_type = data["data_type"]
    run_id = data["run_id"]
    file_name = data["file_name"]
    entries = data["feedback_entries"]

    saved = 0
    for entry in entries:
        try:
            save_feedback(
                data_type=data_type,
                run_id=run_id,
                file_name=file_name,
                timestamp=entry["timestamp"],
                mae_score=entry["mae_score"],
                confidence=entry["confidence"],
                sensor_values=entry.get("sensor_values", {}),
                is_true_anomaly=entry["is_true_anomaly"],
            )
            saved += 1
        except Exception as e:
            print(f"Error saving feedback: {e}")

    return jsonify({"saved": saved, "total": len(entries)})


@app.route("/api/feedback/<data_type>", methods=["GET"])
def get_feedback(data_type):
    """Get all feedback entries for a specific data type."""
    try:
        entries = get_feedback_for_type(data_type)
        # Parse sensor_values back from JSON string
        for entry in entries:
            if isinstance(entry.get("sensor_values"), str):
                entry["sensor_values"] = json.loads(entry["sensor_values"])
        return jsonify({"data_type": data_type, "entries": entries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/summary", methods=["GET"])
def feedback_summary():
    """Get a summary of all feedback across all data types."""
    summary = {}
    for dtype in KNOWN_DATA_TYPES:
        try:
            entries = get_feedback_for_type(dtype)
            true_count = sum(1 for e in entries if e["is_true_anomaly"] == 1)
            false_count = sum(1 for e in entries if e["is_true_anomaly"] == 0)
            summary[dtype] = {
                "total": len(entries),
                "confirmed_anomalies": true_count,
                "false_positives": false_count,
            }
        except Exception:
            summary[dtype] = {"total": 0, "confirmed_anomalies": 0, "false_positives": 0}
    return jsonify(summary)


@app.route("/api/results/<run_id>", methods=["GET"])
def get_results(run_id):
    """Retrieve cached results by run ID."""
    if run_id not in results_store:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(results_store[run_id])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
