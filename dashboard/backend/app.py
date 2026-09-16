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
import hashlib
import pickle
from datetime import datetime, timezone
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
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
import traceback
from dashboard_store import DashboardStore
from measurement_contract import MeasurementRecord
from physical_configuration import ChannelPhysicalConfiguration


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
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DASHBOARD_DB = os.path.join(os.path.dirname(__file__), "dashboard_data.db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
dashboard_store = DashboardStore(DASHBOARD_DB)

# Store run results in memory (keyed by run_id)
results_store = {}

# Known data types (matching folder names in C:\dev\AnomalyDetection\data)
KNOWN_DATA_TYPES = ["biomass", "cataluminescence", "swiss_roll", "default"]


# ====================================================================
# MODEL PERSISTENCE HELPERS
# ====================================================================
def get_model_key(data_type, sensor_cols):
    """Generate a unique key for a model based on data type and feature set.
    
    The key encodes both the data type and a hash of the sorted sensor column
    names so that data with different channel configurations gets a separate
    model even within the same data type.
    """
    features_str = "|".join(sorted(sensor_cols))
    features_hash = hashlib.md5(features_str.encode()).hexdigest()[:8]
    return f"{data_type}_{len(sensor_cols)}ch_{features_hash}"


def get_model_paths(model_key):
    """Return paths for the .pth weights, scaler pickle, and metadata JSON."""
    base = os.path.join(MODELS_DIR, model_key)
    return {
        "weights": f"{base}.pth",
        "scaler": f"{base}_scaler.pkl",
        "meta": f"{base}_meta.json",
    }


def compute_kde_threshold(mae_scores):
    """Compute an anomaly threshold using KDE valley detection.

    Fits a KDE to the MAE score distribution and finds the first local
    minimum (valley) as the natural decision boundary between normal and
    anomalous reconstruction errors.  Falls back to the 97th percentile
    when the distribution is unimodal (no valley exists).

    Returns (threshold, method_used, kde_x, kde_y).
    """
    kde = gaussian_kde(mae_scores)
    x_range = np.linspace(float(np.min(mae_scores)), float(np.max(mae_scores)), 1000)
    kde_values = kde(x_range)

    minima_indices = argrelextrema(kde_values, np.less)[0]
    if len(minima_indices) > 0:
        threshold = float(x_range[minima_indices[0]])
        method = "KDE Local Minimum (Valley)"
    else:
        threshold = float(np.percentile(mae_scores, 97))
        method = "97th Percentile Fallback"

    return threshold, method, x_range.tolist(), kde_values.tolist()


def save_model_artifacts(model, scaler, model_key, sensor_cols,
                         data_type, epoch_losses, used_feedback=False,
                         threshold=None, threshold_method=None):
    """Persist model weights, fitted scaler, and training metadata locally."""
    paths = get_model_paths(model_key)

    # Save PyTorch model weights
    torch.save(model.state_dict(), paths["weights"])

    # Save fitted StandardScaler
    with open(paths["scaler"], "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata (threshold included so inference is stable)
    meta = {
        "model_key": model_key,
        "data_type": data_type,
        "num_features": len(sensor_cols),
        "sensor_columns": sensor_cols,
        "hidden_size": min(64, max(16, len(sensor_cols) * 4)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_epochs": len(epoch_losses),
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "used_feedback": used_feedback,
        "threshold": threshold,
        "threshold_method": threshold_method,
    }
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[MLOps] Saved model artifacts: {model_key} (threshold={threshold:.6f} via {threshold_method})")
    return meta


def load_model_artifacts(model_key, num_features):
    """Load a previously saved model and scaler.
    
    Returns (model, scaler, metadata) or (None, None, None) if not found.
    """
    paths = get_model_paths(model_key)

    if not os.path.exists(paths["weights"]):
        return None, None, None

    try:
        # Reconstruct model architecture and load weights
        model = DynamicLSTMAutoencoder(num_features=num_features)
        model.load_state_dict(torch.load(paths["weights"], weights_only=True))
        model.eval()

        # Load scaler
        with open(paths["scaler"], "rb") as f:
            scaler = pickle.load(f)

        # Load metadata
        meta = {}
        if os.path.exists(paths["meta"]):
            with open(paths["meta"], "r") as f:
                meta = json.load(f)

        print(f"[MLOps] Loaded saved model: {model_key}")
        return model, scaler, meta

    except Exception as e:
        print(f"[MLOps] Failed to load model {model_key}: {e}")
        return None, None, None


def list_saved_models():
    """List all saved models with their metadata."""
    models = []
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith("_meta.json"):
            try:
                with open(os.path.join(MODELS_DIR, fname), "r") as f:
                    meta = json.load(f)
                # Check that weights file still exists
                paths = get_model_paths(meta["model_key"])
                meta["weights_exist"] = os.path.exists(paths["weights"])
                meta["scaler_exists"] = os.path.exists(paths["scaler"])
                models.append(meta)
            except Exception:
                pass
    return models


def delete_model_artifacts(model_key):
    """Delete all artifacts for a given model key."""
    paths = get_model_paths(model_key)
    deleted = []
    for name, path in paths.items():
        if os.path.exists(path):
            os.remove(path)
            deleted.append(name)
    return deleted


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
    model, X_seq_tensor, df_timestamps, df_original, sensor_cols,
    saved_threshold=None
):
    """Detect anomalies using a KDE-derived or persisted threshold.

    If saved_threshold is provided (loaded from _meta.json), it is used
    directly for stable, consistent inference across different files.
    Otherwise, KDE valley detection is run on the current MAE scores.
    """
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

    # KDE curve always computed for visualization
    kde_threshold, kde_method, kde_x, kde_y = compute_kde_threshold(mae_loss)

    # Use saved threshold for stable inference; fall back to KDE on first run
    if saved_threshold is not None:
        threshold = float(saved_threshold)
        threshold_method = "Loaded from saved model"
    else:
        threshold = kde_threshold
        threshold_method = kde_method

    max_mae = float(np.max(mae_loss))

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

    kde_data = {
        "x": kde_x,
        "y": kde_y,
        "threshold": threshold,
        "threshold_method": threshold_method,
        "kde_threshold": kde_threshold,
        "kde_method": kde_method,
    }

    return merged_df, anomalies_only, threshold, mae_loss, per_feature_mae, kde_data


def run_pipeline(file_path, use_feedback=False, force_retrain=False):
    """Execute the full anomaly detection pipeline and return structured results.
    
    MLOps flow:
    - If a saved .pth model exists for this data type + feature set → load it (no training)
    - If no saved model exists → train from scratch and save .pth
    - If force_retrain=True → retrain (incorporating feedback) and overwrite .pth
    """
    file_name = os.path.basename(file_path)

    # Step 1: Load data
    df, sensors, summary = load_and_summarize_data(file_path)
    if len(sensors) == 0:
        raise ValueError("No valid sensor channels detected in the uploaded CSV.")

    # Step 1b: Classify data type
    data_type = classify_data_type(file_path, df, sensors)

    # Step 1c: Extract instrument metadata (if available)
    metadata = extract_instrument_metadata(file_path)

    # Step 2: Compute model key for persistence lookup
    model_key = get_model_key(data_type, sensors)
    model_trained = False
    epoch_losses = []
    model_meta = None

    # Step 3: Try to load a saved model (skip training)
    if not force_retrain:
        saved_model, saved_scaler, model_meta = load_model_artifacts(
            model_key, num_features=len(sensors)
        )
    else:
        saved_model, saved_scaler = None, None

    if saved_model is not None and saved_scaler is not None:
        # ── USE SAVED MODEL (no training) ──
        print(f"[MLOps] Using saved model for inference: {model_key}")
        trained_model = saved_model
        scaler = saved_scaler

        # Scale data using the SAVED scaler (must match training distribution)
        df_scaled = pd.DataFrame(scaler.transform(df[sensors]), columns=sensors)
    else:
        # ── TRAIN NEW MODEL ──
        print(f"[MLOps] No saved model found (or retrain forced). Training: {model_key}")
        model_trained = True

        # Fit a fresh scaler on this data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[sensors]), columns=sensors)

        # Create sequences for training
        TIME_STEPS = 10
        X_seq_train, ts_train = create_sequences(df_scaled, df["Timestamp"], TIME_STEPS)
        X_tensor_train = torch.tensor(X_seq_train, dtype=torch.float32)
        dataset_train = TensorDataset(X_tensor_train)
        train_loader = DataLoader(dataset_train, batch_size=16, shuffle=False)

        model = DynamicLSTMAutoencoder(num_features=len(sensors))
        used_feedback = False

        if (use_feedback or force_retrain) and data_type != "unknown":
            confirmed_normal = get_confirmed_normal_timestamps(data_type)
            if confirmed_normal:
                trained_model, epoch_losses = train_with_feedback(
                    model, train_loader, X_tensor_train, ts_train,
                    confirmed_normal, num_epochs=30
                )
                used_feedback = True
            else:
                trained_model, epoch_losses = train_autoencoder(
                    model, train_loader, num_epochs=30
                )
        else:
            trained_model, epoch_losses = train_autoencoder(
                model, train_loader, num_epochs=30
            )

        # Compute KDE threshold on training data BEFORE saving
        TIME_STEPS_TRAIN = 10
        X_seq_train_tmp, _ = create_sequences(df_scaled, df["Timestamp"], TIME_STEPS_TRAIN)
        X_tensor_train_tmp = torch.tensor(X_seq_train_tmp, dtype=torch.float32)
        trained_model.eval()
        with torch.no_grad():
            recon_tmp = trained_model(X_tensor_train_tmp)
        train_mae = np.mean(
            np.abs(recon_tmp.numpy() - X_tensor_train_tmp.numpy()), axis=(1, 2)
        )
        kde_threshold_saved, kde_method_saved, _, _ = compute_kde_threshold(train_mae)
        print(f"[MLOps] KDE threshold computed: {kde_threshold_saved:.6f} ({kde_method_saved})")

        # Save model artifacts (.pth + scaler + metadata + threshold)
        model_meta = save_model_artifacts(
            trained_model, scaler, model_key, sensors,
            data_type, epoch_losses, used_feedback=used_feedback,
            threshold=kde_threshold_saved, threshold_method=kde_method_saved
        )

    # Step 4: Scale data for inference (already done above) and create sequences
    TIME_STEPS = 10
    X_seq, timestamps_seq = create_sequences(df_scaled, df["Timestamp"], TIME_STEPS)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)

    # Step 5: Detect anomalies — use saved threshold for stable inference
    saved_threshold = model_meta.get("threshold") if model_meta else None
    merged_df, anomalies_df, threshold, mae_scores, per_feature_mae, kde_data = (
        detect_anomalies(trained_model, X_tensor, timestamps_seq, df, sensors,
                         saved_threshold=saved_threshold)
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

    # Build model_info for the response
    model_info = {
        "model_key": model_key,
        "model_trained_this_run": model_trained,
        "loaded_from_cache": not model_trained,
        "trained_at": model_meta.get("trained_at") if model_meta else None,
        "num_epochs": model_meta.get("num_epochs") if model_meta else len(epoch_losses),
        "final_loss": model_meta.get("final_loss") if model_meta else (epoch_losses[-1] if epoch_losses else None),
        "used_feedback": model_meta.get("used_feedback", False) if model_meta else False,
    }

    return {
        "file_name": file_name,
        "data_type": data_type,
        "instrument_metadata": metadata,
        "summary": summary,
        "channels": channels_data,
        "mae_distribution": mae_distribution,
        "kde_data": kde_data,
        "training_loss": training_loss,
        "anomaly_table": anomaly_table,
        "feature_contribution": feature_contribution,
        "channel_statistics": channel_statistics,
        "confidence_distribution": confidence_distribution,
        "total_anomalies": len(anomalies_df),
        "total_points": len(merged_df),
        "anomaly_rate": round(len(anomalies_df) / max(len(merged_df), 1) * 100, 2),
        "existing_feedback": existing_feedback,
        "used_feedback": model_info["used_feedback"],
        "model_info": model_info,
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


# ====================================================================
# EDGE/CLOUD MEASUREMENT API
# ====================================================================
@app.route("/api/measurements", methods=["POST"])
def ingest_measurements():
    """Receive raw edge records without changing their values or flags."""
    payload = request.get_json(silent=True) or {}
    raw_records = payload.get("records", [])
    if not isinstance(raw_records, list) or not raw_records:
        return jsonify({"error": "records must be a non-empty list"}), 400

    try:
        records = [MeasurementRecord.model_validate(item) for item in raw_records]
        stored = dashboard_store.ingest_many(records, run_id=payload.get("run_id"))
        return jsonify({
            "received": len(records),
            "stored": stored,
            "duplicates": len(records) - stored,
        }), 202
    except Exception as exc:
        return jsonify({"error": f"Invalid measurement payload: {exc}"}), 400


@app.route("/api/measurements/<device_id>", methods=["GET"])
def get_measurements(device_id):
    """Return raw measurements and additive quality flags for a device."""
    channel_id = request.args.get("channel_id")
    try:
        records = dashboard_store.list_measurements(device_id, channel_id=channel_id)
        return jsonify({
            "device_id": device_id,
            "records": [record.raw_payload() for record in records],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/measurements/<device_id>/science", methods=["GET"])
def get_science_measurements(device_id):
    """Return the derived science view; raw measurements remain available above."""
    channel_id = request.args.get("channel_id")
    records = dashboard_store.list_science_measurements(
        device_id, channel_id=channel_id
    )
    return jsonify({
        "device_id": device_id,
        "records": [record.raw_payload() for record in records],
        "raw_records_retained": True,
    })


@app.route("/api/diagnostics/<device_id>", methods=["GET"])
def get_device_diagnostics(device_id):
    return jsonify(dashboard_store.diagnostics(device_id))


@app.route("/api/analytics/<device_id>/parameters", methods=["GET"])
def fit_device_parameters(device_id):
    """Fit interpretable channel parameters from stored measurements."""
    return jsonify({
        "device_id": device_id,
        "parameters": dashboard_store.fit_channel_parameters(device_id),
    })


@app.route("/api/edge-config/<device_id>", methods=["GET"])
def get_edge_configuration(device_id):
    config = dashboard_store.get_configuration(
        device_id, request.args.get("config_version")
    )
    if config is None:
        return jsonify({"error": "No configuration found"}), 404
    return jsonify(config)


@app.route("/api/edge-config/<device_id>", methods=["PUT"])
def save_edge_configuration(device_id):
    payload = request.get_json(silent=True) or {}
    required = [
        "config_version",
        "ruleset_version",
        "parameter_set_version",
        "configuration",
    ]
    missing = [field for field in required if field not in payload]
    if missing or not isinstance(payload.get("configuration"), dict):
        return jsonify({
            "error": "Missing required configuration fields",
            "missing": missing,
        }), 400
    dashboard_store.save_configuration(
        device_id=device_id,
        config_version=payload["config_version"],
        ruleset_version=payload["ruleset_version"],
        parameter_set_version=payload["parameter_set_version"],
        configuration=payload["configuration"],
    )
    return jsonify(dashboard_store.get_configuration(
        device_id, payload["config_version"]
    )), 201


@app.route("/api/devices/<device_id>/channels/<channel_id>/physical-config", methods=["PUT"])
def save_channel_physical_configuration(device_id, channel_id):
    """Validate and publish one channel's measured physical configuration."""
    payload = request.get_json(silent=True) or {}
    payload["device_id"] = device_id
    payload["channel_id"] = channel_id
    try:
        configuration = ChannelPhysicalConfiguration.model_validate(payload)
        dashboard_store.save_channel_configuration(configuration)
        return jsonify(configuration.as_edge_payload()), 201
    except Exception as exc:
        return jsonify({"error": f"Invalid channel configuration: {exc}"}), 400


@app.route("/api/devices/<device_id>/channels/<channel_id>/physical-config", methods=["GET"])
def get_channel_physical_configuration(device_id, channel_id):
    configuration = dashboard_store.get_channel_configuration(
        device_id, channel_id, request.args.get("config_version")
    )
    if configuration is None:
        return jsonify({"error": "Channel configuration not found"}), 404
    return jsonify(configuration.as_edge_payload())


@app.route("/api/devices/<device_id>/physical-config", methods=["GET"])
def list_device_physical_configurations(device_id):
    configurations = dashboard_store.list_channel_configurations(device_id)
    return jsonify({
        "device_id": device_id,
        "channels": [configuration.as_edge_payload() for configuration in configurations],
    })


# ====================================================================
# MODEL MANAGEMENT API ROUTES
# ====================================================================
@app.route("/api/retrain", methods=["POST"])
def retrain_model():
    """Force retrain the model for a given CSV, incorporating latest feedback.
    
    This deletes the cached .pth and retrains from scratch with feedback-aware
    training (upweighting confirmed-normal timestamps).
    """
    data = request.get_json()
    file_path = data.get("path")

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400

    try:
        results = sanitize(
            run_pipeline(file_path, use_feedback=True, force_retrain=True)
        )
        run_id = str(uuid.uuid4())
        results_store[run_id] = results
        return jsonify({
            "run_id": run_id,
            "results": results,
            "message": "Model retrained successfully with latest feedback",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_models():
    """List all saved model artifacts with their metadata."""
    try:
        models = list_saved_models()
        return jsonify({"models": models, "models_dir": MODELS_DIR})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<model_key>", methods=["DELETE"])
def delete_model(model_key):
    """Delete a saved model (forces retrain on next run)."""
    try:
        deleted = delete_model_artifacts(model_key)
        if deleted:
            return jsonify({
                "message": f"Deleted model artifacts: {', '.join(deleted)}",
                "model_key": model_key,
            })
        else:
            return jsonify({"error": "No artifacts found for this model key"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
