# Anomaly Detection Dashboard

An advanced machine learning dashboard for unsupervised anomaly detection in time-series sensor data, with human-in-the-loop feedback that improves model accuracy over time.

This dashboard faithfully reproduces the PyTorch LSTM Autoencoder models from the research notebooks (Biomass, Cataluminescence, Swiss, Default datasets) while providing a responsive, real-time interface powered by background task processing and persistent feedback storage.

## Features

- **LSTM Autoencoder Engine**: Trains a PyTorch LSTM model on-the-fly across all sensor channels simultaneously. Architecture, hyperparameters, and thresholding match the original research notebooks exactly.
- **Asynchronous Task Queue**: Long-running ML training jobs run in background threads (mirrors Celery/Redis pattern). Frontend polls for live progress with epoch-level updates.
- **Human-in-the-Loop Feedback**: Users can confirm or reject detected anomalies. Feedback is persisted in SQLite and incorporated into subsequent training runs — confirmed anomalies adjust the threshold downward, rejected false positives are force-unflagged.
- **Feedback-Aware Retraining**: When feedback labels exist for a dataset, the next LSTM run automatically queries them and adjusts anomaly detection accordingly.
- **Training Run History**: Every training run is recorded in the database with full metrics (epochs, MAE, threshold, anomaly count, feedback labels used).
- **Interactive Visualizations**: High-performance Plotly charts for data exploration, confidence intervals, and anomaly overlay.
- **Robust CSV Parsing**: Automatically detects and ingests both cleaned CSV formats and raw DAQ instrument exports.

## Project Structure

```
IITM_project/
├── app.py              # Flask backend — routes, LSTM training, task queue
├── feedback_db.py      # SQLite database module for feedback labels & training history
├── feedback.db         # Auto-created SQLite database (persistent storage)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Dashboard HTML
├── static/
│   ├── css/style.css   # Dashboard styling
│   └── js/app.js       # Frontend logic — polling, charts, feedback UI
├── uploads/            # Experiment data (CSV files)
│   ├── Biomass/
│   ├── cataluminiscence/
│   ├── Swiss Roll/
│   └── default/
└── *.ipynb             # Research notebooks (reference implementations)
```

## Installation

1. Clone or download the repository.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note**: `torch` (PyTorch) is required for the LSTM features. If unavailable, the dashboard will still launch but LSTM capabilities will be disabled.

## Running the Application

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

## Architecture

### Backend (Python/Flask)

```
Browser → POST /api/run_lstm → Flask spawns background thread → returns {task_id}
Browser → polls GET /api/task/{id} every 2s → {status, progress%, message}
Thread  → trains LSTM → queries feedback DB → adjusts threshold → saves results
```

- **Task Queue**: `threading.Thread` with a `TASKS` dict (mirrors Celery pattern)
- **Feedback DB**: SQLite with WAL mode for concurrent read/write safety
- **ML Pipeline**: PyTorch LSTM Autoencoder with StandardScaler, L1Loss, 97th-percentile thresholding

### Frontend (Vanilla JS)

- **Plotly.js** for interactive time-series charts
- **Async polling** with live progress bar during training
- **Feedback UI** with per-anomaly confirm/reject buttons and bulk operations

## Usage

1. **Upload Data**: Upload a raw DAQ export or cleaned CSV file.
2. **Select Dataset**: Choose the experiment and file from the sidebar.
3. **Train Model**: Configure epochs and window size, click "Run LSTM Detection".
4. **Review Anomalies**: Examine detected anomalies in the results table.
5. **Provide Feedback**: Click ✓ Anomaly or ✗ Normal on each detected point, then "Save All Feedback".
6. **Retrain**: Run LSTM again — the model now uses your feedback to adjust its threshold and flagging.
