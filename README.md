# Anomaly Detection Dashboard

A full-stack application for anomaly detection in sensor data using an LSTM Autoencoder or a Physics Constrained with Isolation Forest pipeline.

## Overview

This project provides an interactive dashboard to upload sensor data (CSV), run either the PyTorch-based LSTM Autoencoder or the Physics Constrained with Isolation Forest model, and review the results. Both models support saved human feedback, with a workflow suited to each model.

## Key Features

- **MLOps & Model Persistence:** Models and data scalers are saved locally after training. Subsequent runs on the same instrument data type load the saved PyTorch `.pth` weights for instant inference (no redundant training).
- **Physics + Isolation Forest Run Tracking:** Each hybrid-model upload records a versioned, reproducible training run, data SHA-256, feature schema, parameters, package versions, fit/score timings, anomaly metrics, and a serialized model artifact. Runs are stored in the local model registry and SQLite tracking database.
- **Hybrid Model Human Feedback:** Review Isolation Forest and physics flags as true anomalies or false positives, save the labels, and apply them on a repeat run. Matching file contents use exact reviewed-point overrides; after at least three examples of each class across two other files, a supervised Random Forest feedback layer can correct predictions on new files.
- **LSTM Autoencoder Pipeline:** Dynamically scales hidden layers based on the number of input features. Trains on uploaded data to learn normal behavior and flags sequences with high reconstruction error (MAE).
- **Automatic Data Classification:** Intelligently parses CSV headers to identify the data source (Biomass, Cataluminescence, Swiss Roll, or Default 34970A instruments).
- **Instrument Metadata Extraction:** Automatically extracts and displays hardware metadata (Model, Serial Number, Firmware, Channel Configurations) from Agilent/Keysight log files.
- **LSTM Human Feedback:** Users can mark detected anomalies as "Yes" (true anomaly) or "No" (false positive). Feedback is persisted locally and can be included in LSTM retraining using a weighted loss.
- **Physics + Isolation Forest Human Feedback:** Users can label flagged readings and save them locally. On a later run, saved labels can override predictions for the same file contents; with enough labels from other files, a supervised Random Forest feedback layer can generalize corrections to new files.
- **Rich Interactive Visualizations:**
  - Per-channel time-series charts with anomaly overlays.
  - Model diagnostic charts: Training Loss curve and Reconstruction Error (MAE) distribution.
  - Anomaly Confidence distribution.
  - Per-feature error contribution bar charts.
- **Advanced Statistical Analysis:** Computes and visualizes per-channel Mean, Std Dev, Min, Max, Skewness, Kurtosis, and 95% Confidence Intervals.

## Tech Stack

### Backend (Python)
- **Flask:** REST API server.
- **PyTorch:** Deep learning framework for the LSTM Autoencoder.
- **Pandas & NumPy:** Data manipulation, preprocessing, and statistical calculations.
- **Scikit-Learn:** Data scaling (StandardScaler).
- **SQLite:** Lightweight, local database for storing user feedback per data type.

### Frontend (React)
- **Vite:** Fast frontend build tool.
- **React.js:** Component-based UI framework.
- **Recharts:** Composable charting library for React.
- **Vanilla CSS:** Custom, modern, dark-themed design system (no external CSS frameworks).

## Model Architecture

The core anomaly detection engine relies on a PyTorch-based **LSTM Autoencoder** designed for multivariate time-series data. The project contains multiple Jupyter Notebooks (`*LSTM_autoencoder.ipynb`) demonstrating the model on specific datasets (Biomass, Cataluminescence, Swiss Roll). The dashboard uses a unified, dynamic version:

- **DynamicLSTMAutoencoder:** 
  - **Encoder:** An LSTM layer that compresses the input sequence of `num_features` into a bottleneck latent representation (`hidden_size`).
  - **Hidden Size:** Dynamically scaled based on input features (`min(64, max(16, num_features * 4))`).
  - **Decoder:** An LSTM layer that takes the latent representation and attempts to reconstruct the original time-series sequence.
  - **Sliding Window:** Processes data in time-step sequences (e.g., 10 time steps per window).
  - **Anomaly Scoring:** The Mean Absolute Error (MAE) between the input sequence and the reconstructed sequence serves as the anomaly score. High MAE indicates the model hasn't seen this pattern before, flagging it as an anomaly.

The **Physics Constrained with Isolation Forest** option follows a separate batch pipeline based on `new_15_iso_phy_modelling.ipynb`: it parses and validates the DAQ export, applies the K-type physical veto, engineers temperature and raw `dT/dt` features, fits and scores a deterministic Isolation Forest, validates the outputs, then registers the versioned artifact and run metadata. Because this notebook defines a per-file unsupervised fit and there is no separate labeled training corpus, the model is trained on each uploaded file; fast completion on a few thousand rows is expected. It does not add an artificial delay. Each successful run is visible in the result panel and through `GET /api/mlops/runs` or `GET /api/mlops/runs/<run_id>`.

### Human Feedback Workflows

- **LSTM Autoencoder:** In the Anomaly Log, label flagged readings with **Yes** or **No**, then select **Save Feedback**. Use **Retrain with feedback** to run the LSTM again with saved labels included in its weighted training loss.
- **Physics Constrained with Isolation Forest:** Label flagged readings with **Yes** or **No** and select **Save Feedback**. Enable **Use past feedback** when running the model again (or select **Re-run with saved feedback** from the results). For identical CSV contents, saved labels are applied to their reviewed readings. For different files, a Random Forest feedback layer is eligible once there are at least three examples of each label across at least two other file contents; it only corrects physics-valid readings when its confidence is at least 80%. This feedback layer supplements the per-file Isolation Forest; it does not retrain the Isolation Forest itself.

## Architecture

```
dashboard/
├── backend/
│   ├── app.py                 # Core Flask API, PyTorch model, and data processing
│   ├── requirements.txt       # Python dependencies
│   ├── uploads/               # Temporary storage for uploaded CSVs
│   ├── feedback_db/           # SQLite databases (auto-generated)
│   └── models/                # Saved PyTorch .pth weights and .pkl scalers
└── frontend/
    ├── package.json           # Node dependencies
    ├── index.html             # Entry HTML
    ├── vite.config.js         # Vite configuration
    └── src/
        ├── App.jsx            # Main application state and layout
        ├── api.js             # API client functions
        ├── index.css          # Global styles and design tokens
        ├── main.jsx           # React DOM entry point
        └── components/        # React components (Charts, Tables, Panels)
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm (or yarn/pnpm)

### 1. Backend Setup

Open a terminal and navigate to the `backend` directory:
```bash
cd dashboard\backend
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```
*(Note: It is recommended to use a virtual environment).*

Start the Flask server:
```bash
python app.py
```
The backend will run on `http://127.0.0.1:5000`.

### 2. Frontend Setup

Open a new terminal and navigate to the `frontend` directory:
```bash
cd dashboard\frontend
```

Install the required Node packages:
```bash
npm install
```

Start the Vite development server:
```bash
npm run dev
```
The frontend will run on `http://localhost:5173`.

## Usage Guide

1. **Open the Dashboard:** Navigate to `http://localhost:5173` in your web browser.
2. **Upload Data:** Drag and drop or click to upload a supported CSV file.
3. **Run Analysis:** Choose a model and click **"Run Model"**. The backend runs the selected model and returns its results.
4. **Explore Results:**
   - **Overview:** View time-series plots for all channels, training loss, and error distributions.
   - **Sensor Channels:** Drill down into specific channels and view detailed statistics.
   - **Model Diagnostics:** Review the model's configuration and reconstruction error breakdowns.
   - **Anomaly Log:** View the paginated table of all detected anomalies.
   - **Statistics:** Analyze skewness, kurtosis, and confidence intervals across all sensors.
5. **Provide Feedback:** For the LSTM model, use the **Anomaly Log** tab to label readings and save them; use **"Retrain with feedback"** to include them in LSTM training. For Physics + Isolation Forest, label flagged readings in its results and save them; enable **"Use past feedback"** on the next run or use **"Re-run with saved feedback"** to apply them.
