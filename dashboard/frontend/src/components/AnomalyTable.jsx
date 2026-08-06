import { useState, useMemo } from 'react';
import { submitFeedback } from '../api';

export default function AnomalyTable({
  anomalies,
  sensorColumns,
  dataType,
  fileName,
  runId,
  existingFeedback = {},
}) {
  const [sortField, setSortField] = useState('confidence');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(0);
  const [localFeedback, setLocalFeedback] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitMsg, setSubmitMsg] = useState(null);
  const perPage = 25;

  // Merge existing (from DB) and local (unsaved) feedback
  const mergedFeedback = useMemo(() => {
    const merged = {};
    for (const [ts, val] of Object.entries(existingFeedback)) {
      merged[ts] = val === 1;
    }
    for (const [ts, val] of Object.entries(localFeedback)) {
      merged[ts] = val;
    }
    return merged;
  }, [existingFeedback, localFeedback]);

  const sorted = useMemo(() => {
    const arr = [...anomalies];
    arr.sort((a, b) => {
      let va, vb;
      if (sortField === 'confidence') {
        va = a.confidence;
        vb = b.confidence;
      } else if (sortField === 'timestamp') {
        va = a.timestamp;
        vb = b.timestamp;
      } else if (sortField === 'mae_score') {
        va = a.mae_score;
        vb = b.mae_score;
      } else {
        va = a.sensor_values[sortField] || 0;
        vb = b.sensor_values[sortField] || 0;
      }
      if (sortDir === 'asc') return va > vb ? 1 : -1;
      return va < vb ? 1 : -1;
    });
    return arr;
  }, [anomalies, sortField, sortDir]);

  const totalPages = Math.ceil(sorted.length / perPage);
  const pageData = sorted.slice(page * perPage, (page + 1) * perPage);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir('desc');
    }
    setPage(0);
  };

  const handleFeedback = (timestamp, isTrueAnomaly) => {
    setLocalFeedback((prev) => ({ ...prev, [timestamp]: isTrueAnomaly }));
  };

  const handleSubmitAll = async () => {
    const entries = Object.entries(localFeedback).map(([ts, isTrue]) => {
      const anomaly = anomalies.find((a) => a.timestamp === ts);
      return {
        timestamp: ts,
        mae_score: anomaly?.mae_score || 0,
        confidence: anomaly?.confidence || 0,
        sensor_values: anomaly?.sensor_values || {},
        is_true_anomaly: isTrue,
      };
    });

    if (entries.length === 0) return;

    setSubmitting(true);
    setSubmitMsg(null);
    try {
      const result = await submitFeedback(runId, dataType, fileName, entries);
      setSubmitMsg(`Saved ${result.saved} feedback entries.`);
      // Clear local feedback since it's now persisted
      setLocalFeedback({});
    } catch (err) {
      setSubmitMsg(`Error: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const pendingCount = Object.keys(localFeedback).length;

  const getConfidenceColor = (c) => {
    if (c >= 80) return '#ef4444';
    if (c >= 60) return '#f59e0b';
    return '#3b82f6';
  };

  const SortArrow = ({ field }) => {
    if (sortField !== field) return null;
    return <span style={{ marginLeft: 4 }}>{sortDir === 'asc' ? '\u2191' : '\u2193'}</span>;
  };

  if (anomalies.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '48px' }}>
        <p style={{ color: 'var(--text-muted)' }}>No anomalies detected in this dataset.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card__header">
        <span className="card__title">Detected Anomalies ({anomalies.length})</span>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {submitMsg && (
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-green)' }}>
              {submitMsg}
            </span>
          )}
          {pendingCount > 0 && (
            <button
              className="btn btn--primary"
              style={{ padding: '6px 14px', fontSize: '0.78rem' }}
              onClick={handleSubmitAll}
              disabled={submitting}
            >
              {submitting ? 'Saving...' : `Save Feedback (${pendingCount})`}
            </button>
          )}
          {totalPages > 1 && (
            <>
              <button
                className="btn btn--ghost"
                style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Prev
              </button>
              <span
                style={{
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {page + 1}/{totalPages}
              </span>
              <button
                className="btn btn--ghost"
                style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </>
          )}
        </div>
      </div>

      {/* Feedback legend */}
      <div
        style={{
          display: 'flex',
          gap: '16px',
          marginBottom: '12px',
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
        }}
      >
        <span>
          Verify each anomaly: mark as
          <span style={{ color: 'var(--accent-red)', fontWeight: 500 }}> Yes</span> (true anomaly) or
          <span style={{ color: 'var(--accent-green)', fontWeight: 500 }}> No</span> (false positive),
          then save.
        </span>
      </div>

      <div className="anomaly-table-wrapper">
        <table className="anomaly-table">
          <thead>
            <tr>
              <th style={{ width: '90px' }}>Verify</th>
              <th onClick={() => handleSort('timestamp')} style={{ cursor: 'pointer' }}>
                Timestamp
                <SortArrow field="timestamp" />
              </th>
              <th onClick={() => handleSort('confidence')} style={{ cursor: 'pointer' }}>
                Confidence
                <SortArrow field="confidence" />
              </th>
              <th onClick={() => handleSort('mae_score')} style={{ cursor: 'pointer' }}>
                MAE Score
                <SortArrow field="mae_score" />
              </th>
              {sensorColumns.map((col) => (
                <th key={col} onClick={() => handleSort(col)} style={{ cursor: 'pointer' }}>
                  {col}
                  <SortArrow field={col} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageData.map((row, i) => {
              const fb = mergedFeedback[row.timestamp];
              const isSaved = row.timestamp in existingFeedback;
              const isLocal = row.timestamp in localFeedback;

              return (
                <tr key={i}>
                  <td>
                    <div className="feedback-buttons">
                      <button
                        className={`feedback-btn feedback-btn--yes ${fb === true ? 'feedback-btn--active' : ''}`}
                        onClick={() => handleFeedback(row.timestamp, true)}
                        title="True anomaly"
                        disabled={isSaved && !isLocal}
                      >
                        Yes
                      </button>
                      <button
                        className={`feedback-btn feedback-btn--no ${fb === false ? 'feedback-btn--active' : ''}`}
                        onClick={() => handleFeedback(row.timestamp, false)}
                        title="False positive"
                        disabled={isSaved && !isLocal}
                      >
                        No
                      </button>
                      {isSaved && !isLocal && (
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginLeft: 2 }}>
                          saved
                        </span>
                      )}
                    </div>
                  </td>
                  <td>{row.timestamp}</td>
                  <td>
                    <div className="confidence-bar">
                      <div className="confidence-bar__track">
                        <div
                          className="confidence-bar__fill"
                          style={{
                            width: `${row.confidence}%`,
                            background: getConfidenceColor(row.confidence),
                          }}
                        />
                      </div>
                      <span
                        className="confidence-bar__label"
                        style={{ color: getConfidenceColor(row.confidence) }}
                      >
                        {row.confidence}%
                      </span>
                    </div>
                  </td>
                  <td>{row.mae_score}</td>
                  {sensorColumns.map((col) => (
                    <td key={col}>{row.sensor_values[col]}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
