import { useState } from 'react';
import { uploadCSV, runModel } from './api';
import UploadZone from './components/UploadZone';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [filePath, setFilePath] = useState(null);
  const [results, setResults] = useState(null);
  const [runId, setRunId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useFeedback, setUseFeedback] = useState(false);

  const handleFileSelect = async (file) => {
    setError(null);
    setResults(null);
    setRunId(null);
    try {
      const data = await uploadCSV(file);
      setUploadedFile({ name: file.name, id: data.file_id });
      setFilePath(data.path);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRun = async () => {
    if (!filePath) return;
    setError(null);
    setLoading(true);
    try {
      const data = await runModel(filePath, useFeedback);
      setResults(data.results);
      setRunId(data.run_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRerun = async () => {
    if (!filePath) return;
    setError(null);
    setLoading(true);
    setResults(null);
    try {
      const data = await runModel(filePath, true);
      setResults(data.results);
      setRunId(data.run_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setUploadedFile(null);
    setFilePath(null);
    setResults(null);
    setRunId(null);
    setError(null);
    setLoading(false);
    setUseFeedback(false);
  };

  const dataTypeLabel = (dt) => {
    const map = {
      biomass: 'Biomass',
      cataluminescence: 'Cataluminescence',
      swiss_roll: 'Swiss Roll',
      default: 'Default (34970A)',
      unknown: 'Unknown',
    };
    return map[dt] || dt;
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="app-header__title-group">
          <h1 className="app-header__title">Anomaly Detection</h1>
          <p className="app-header__subtitle">
            LSTM Autoencoder Pipeline
            {results && results.data_type && (
              <span style={{ marginLeft: 12, color: 'var(--accent-blue)' }}>
                / {dataTypeLabel(results.data_type)}
              </span>
            )}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {uploadedFile && (
            <div className="file-chip">
              <span>{uploadedFile.name}</span>
              <button className="file-chip__remove" onClick={handleReset}>
                x
              </button>
            </div>
          )}

          {uploadedFile && !results && !loading && (
            <>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '0.8rem',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={useFeedback}
                  onChange={(e) => setUseFeedback(e.target.checked)}
                  style={{ accentColor: 'var(--accent-blue)' }}
                />
                Use past feedback
              </label>
              <button className="btn btn--primary" onClick={handleRun}>
                Run Model
              </button>
            </>
          )}

          {results && (
            <button className="btn btn--ghost" onClick={handleReset}>
              New Analysis
            </button>
          )}
        </div>
      </header>

      {error && (
        <div
          className="card"
          style={{
            borderColor: 'var(--accent-red)',
            marginBottom: '20px',
            padding: '14px 20px',
          }}
        >
          <span style={{ color: 'var(--accent-red)', fontSize: '0.85rem' }}>
            {error}
          </span>
        </div>
      )}

      {!uploadedFile && !loading && !results && (
        <UploadZone onFileSelect={handleFileSelect} />
      )}

      {loading && (
        <div className="progress-overlay">
          <div className="spinner" />
          <p className="progress-overlay__text">
            Training LSTM Autoencoder and detecting anomalies...
          </p>
          <p className="progress-overlay__subtext">
            This runs the actual model — it may take a moment depending on data size.
          </p>
        </div>
      )}

      {results && (
        <ResultsDashboard
          data={results}
          runId={runId}
          onRerun={handleRerun}
        />
      )}
    </div>
  );
}

export default App;
