# Anomaly Detection Dashboard

A full-stack application for real-time anomaly detection in multi-channel sensor data using a Deep Learning (LSTM Autoencoder) pipeline.

## Overview

This project provides a robust, interactive dashboard to upload sensor data (CSV), automatically classify the instrument type, run a PyTorch-based LSTM Autoencoder to detect anomalies, and visualize the results. It features a Human-in-the-Loop (HITL) feedback system, allowing users to verify anomalies and retrain the model based on their feedback.

## Key Features

- **LSTM Autoencoder Pipeline:** Dynamically scales hidden layers based on the number of input features. Trains on uploaded data to learn normal behavior and flags sequences with high reconstruction error (MAE).
- **Automatic Data Classification:** Intelligently parses CSV headers to identify the data source (Biomass, Cataluminescence, Swiss Roll, or Default 34970A instruments).
- **Instrument Metadata Extraction:** Automatically extracts and displays hardware metadata (Model, Serial Number, Firmware, Channel Configurations) from Agilent/Keysight log files.
- **Human-in-the-Loop Feedback:** Users can review detected anomalies in a paginated table and mark them as "Yes" (True Anomaly) or "No" (False Positive). Feedback is persisted locally in SQLite databases.
- **Feedback-Aware Retraining:** The model can be retrained using past user feedback, applying weighted loss to reduce false positives on confirmed normal data points.
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

## Architecture

```
dashboard/
├── backend/
│   ├── app.py                 # Core Flask API, PyTorch model, and data processing
│   ├── requirements.txt       # Python dependencies
│   ├── uploads/               # Temporary storage for uploaded CSVs
│   └── feedback_db/           # SQLite databases (auto-generated)
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
cd c:\dev\AnomalyDetection\dashboard\backend
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
cd c:\dev\AnomalyDetection\dashboard\frontend
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
3. **Run Analysis:** Click the **"Run Model"** button. The backend will parse the file, train the LSTM model, detect anomalies, and return the results.
4. **Explore Results:**
   - **Overview:** View time-series plots for all channels, training loss, and error distributions.
   - **Sensor Channels:** Drill down into specific channels and view detailed statistics.
   - **Model Diagnostics:** Review the model's configuration and reconstruction error breakdowns.
   - **Anomaly Log:** View the paginated table of all detected anomalies.
   - **Statistics:** Analyze skewness, kurtosis, and confidence intervals across all sensors.
5. **Provide Feedback:** In the **Anomaly Log** tab, use the **Yes/No** buttons to verify anomalies. Click **"Save Feedback"** to persist your classifications.
6. **Retrain:** Click **"Retrain with feedback"** to run the model again, incorporating your saved feedback to improve accuracy.
