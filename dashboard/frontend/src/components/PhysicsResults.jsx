import { useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { submitFeedback } from '../api';

const formatTime = (value) => {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

function TimeSeriesChart({ title, description, data, children }) {
  return <section className="card" style={{ marginBottom: 20 }}>
    <h3 style={{ marginTop: 0 }}>{title}</h3>
    <p style={{ color: 'var(--text-secondary)', marginTop: -8 }}>{description}</p>
    <div style={{ width: '100%', height: 380 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 24, left: 8, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
          <XAxis
            dataKey="timestamp"
            type="category"
            tickFormatter={formatTime}
            minTickGap={48}
            height={48}
            label={{ value: 'Time', position: 'insideBottom', offset: -12 }}
          />
          <YAxis
            domain={['auto', 'auto']}
            label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            labelFormatter={(value) => value}
            formatter={(value, name) => [`${Number(value).toFixed(3)} °C`, name]}
          />
          <Legend />
          {children}
        </LineChart>
      </ResponsiveContainer>
    </div>
  </section>;
}

function PhysicsVetoDot({ cx, cy }) {
  if (cx == null || cy == null) return null;
  return <path d={`M${cx - 4},${cy - 4} L${cx + 4},${cy + 4} M${cx + 4},${cy - 4} L${cx - 4},${cy + 4}`} stroke="#dc2626" strokeWidth={2} />;
}

export default function PhysicsResults({ data, runId, onRerun }) {
  const [localFeedback, setLocalFeedback] = useState({});
  const [savedFeedback, setSavedFeedback] = useState(data.existing_feedback || {});
  const [savingFeedback, setSavingFeedback] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState(null);
  const chartData = useMemo(() => data.points.map((point) => ({
    ...point,
    physicsValidTemperature: point.physics_flag === 'VALID' ? point.temperature : null,
    hybridNormalTemperature: point.final_anomaly === false ? point.temperature : null,
    statisticalTemperature: point.final_anomaly && point.physics_flag === 'VALID' ? point.temperature : null,
    physicsTemperature: point.final_anomaly && point.physics_flag === 'IMPLAUSIBLE' ? point.temperature : null,
  })), [data.points]);
  const anomalies = data.points.filter((point) => point.model_quality !== 'VALID' || point.final_anomaly);
  const pendingCount = Object.keys(localFeedback).length;

  const handleSaveFeedback = async () => {
    const entries = Object.entries(localFeedback).map(([timestamp, isTrueAnomaly]) => {
      const point = anomalies.find((reading) => reading.timestamp === timestamp);
      return {
        timestamp,
        mae_score: point?.isolation_forest_score ?? 0,
        confidence: 0,
        sensor_values: {
          temperature: point?.temperature,
          dT_dt_raw: point?.dT_dt_raw,
          isolation_forest_score: point?.isolation_forest_score,
          physics_flag: point?.physics_flag,
          dataset_sha256: data.mlops?.source_sha256,
        },
        is_true_anomaly: isTrueAnomaly,
      };
    });
    setSavingFeedback(true);
    setFeedbackMessage(null);
    try {
      const result = await submitFeedback(runId, data.feedback_data_type, data.file_name, entries);
      setSavedFeedback((previous) => ({ ...previous, ...localFeedback }));
      setLocalFeedback({});
      setFeedbackMessage(`Saved ${result.saved} review${result.saved === 1 ? '' : 's'}. Use “Use past feedback” or rerun below to apply them.`);
    } catch (error) {
      setFeedbackMessage(`Could not save feedback: ${error.message}`);
    } finally {
      setSavingFeedback(false);
    }
  };

  return <div className="fade-in">
    {data.mlops && <section className="card" style={{ marginBottom: 20 }}>
      <h3 style={{ marginTop: 0 }}>Model run</h3>
      <p style={{ color: 'var(--text-secondary)' }}>Status: {data.mlops.status} · Version: {data.mlops.model_version} · Run ID: {data.run_id}</p>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
        Trained per upload on {data.mlops.training_rows.toLocaleString()} rows · Fit {data.mlops.timing_seconds.fit.toFixed(3)}s · Score {data.mlops.timing_seconds.predict_and_score.toFixed(3)}s · Data SHA-256 {data.mlops.source_sha256.slice(0, 16)}…
      </p>
      {data.used_feedback && <p style={{ color: 'var(--accent-green)', fontSize: '0.82rem' }}>
        Past feedback applied to {data.mlops.feedback?.applied_to_points ?? 0} points ({data.mlops.feedback?.changed_model_decisions ?? 0} changed the model output).
      </p>}
    </section>}

    <section className="card" style={{ marginBottom: 20 }}>
      <h2 style={{ marginTop: 0 }}>{data.model_name || 'Physics Constrained with Isolation Forest'}</h2>
      <p style={{ color: 'var(--text-secondary)' }}>Temperature channel: {data.channel} · {data.file_name}</p>
      <div className="stats-row">
        <div className="card stat-card"><p className="stat-card__label">Points processed</p><p className="stat-card__value">{data.total_points.toLocaleString()}</p></div>
        <div className="card stat-card"><p className="stat-card__label">Total flagged</p><p className="stat-card__value">{data.total_anomalies.toLocaleString()}</p></div>
        <div className="card stat-card"><p className="stat-card__label">Physics vetoes (model)</p><p className="stat-card__value" style={{ color: 'var(--accent-red)' }}>{(data.physics_anomalies ?? data.total_anomalies).toLocaleString()}</p></div>
        <div className="card stat-card"><p className="stat-card__label">Isolation Forest outliers (model)</p><p className="stat-card__value" style={{ color: '#d97706' }}>{(data.isolation_forest_anomalies ?? 0).toLocaleString()}</p></div>
        <div className="card stat-card"><p className="stat-card__label">Anomaly rate</p><p className="stat-card__value">{data.anomaly_rate}%</p></div>
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
        Limits: {data.config.sensor_min_c}–{data.config.sensor_max_c} °C sensor range, {data.config.sink_c}–{data.config.drive_max_c} °C environment, τ={data.config.tau_seconds}s, margin={data.config.tau_uncertainty_margin}.
      </p>
    </section>

    <TimeSeriesChart
      title="Physics-based anomaly detection"
      description="Valid readings and points vetoed by the K-type probe physical constraints."
      data={chartData}
    >
      <Line dataKey="temperature" name="Raw trajectory" stroke="#d1d5db" strokeWidth={1.5} dot={false} isAnimationActive={false} />
      <Line dataKey="physicsValidTemperature" name="Valid (obeys physics)" stroke="none" dot={{ r: 2.5, fill: '#2563eb', stroke: '#2563eb' }} activeDot={false} isAnimationActive={false} />
      <Line dataKey="physicsTemperature" name="IMPLAUSIBLE (vetoed)" stroke="none" dot={<PhysicsVetoDot />} activeDot={false} isAnimationActive={false} />
    </TimeSeriesChart>

    <TimeSeriesChart
      title="Hybrid anomaly detection: Physics + Isolation Forest"
      description="Physics vetoes take precedence; Isolation Forest outliers are shown among physics-valid readings."
      data={chartData}
    >
      <Line dataKey="temperature" name="Raw trajectory" stroke="#d1d5db" strokeWidth={1.5} dot={false} isAnimationActive={false} />
      <Line dataKey="hybridNormalTemperature" name="Normal operation" stroke="none" dot={{ r: 2.5, fill: '#2563eb', stroke: '#2563eb' }} activeDot={false} isAnimationActive={false} />
      <Line dataKey="statisticalTemperature" name="Isolation Forest outlier" stroke="none" dot={{ r: 4, fill: '#f08c00', stroke: '#c76c00' }} activeDot={false} isAnimationActive={false} />
      <Line dataKey="physicsTemperature" name="Physics veto (impossible)" stroke="none" dot={<PhysicsVetoDot />} activeDot={false} isAnimationActive={false} />
    </TimeSeriesChart>

    <section className="card">
      <div className="card__header" style={{ alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span className="card__title">Flagged readings for review ({anomalies.length})</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          {feedbackMessage && <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem' }}>{feedbackMessage}</span>}
          {pendingCount > 0 && <button className="btn btn--primary" style={{ padding: '6px 14px', fontSize: '0.78rem' }} onClick={handleSaveFeedback} disabled={savingFeedback}>
            {savingFeedback ? 'Saving...' : `Save Feedback (${pendingCount})`}
          </button>}
          {Object.keys(savedFeedback).length > 0 && <button className="btn btn--ghost" style={{ padding: '6px 14px', fontSize: '0.78rem' }} onClick={onRerun}>
            Re-run with saved feedback
          </button>}
        </div>
      </div>
      {anomalies.length === 0 ? <p style={{ color: 'var(--text-muted)' }}>No anomalies detected.</p> :
        <div className="anomaly-table-wrapper"><table className="anomaly-table"><thead><tr><th>Review</th><th>Timestamp</th><th>Temperature (°C)</th><th>Classification</th><th>Isolation Forest score</th><th>Reason</th></tr></thead><tbody>
          {anomalies.map((point, index) => <tr key={`${point.timestamp}-${index}`}>
            <td><div className="feedback-buttons">
              {(() => {
                const hasLocal = Object.prototype.hasOwnProperty.call(localFeedback, point.timestamp);
                const hasSaved = Object.prototype.hasOwnProperty.call(savedFeedback, point.timestamp);
                const selected = hasLocal ? localFeedback[point.timestamp] : savedFeedback[point.timestamp];
                return <>
                  <button className={`feedback-btn feedback-btn--yes ${selected === true ? 'feedback-btn--active' : ''}`} onClick={() => setLocalFeedback((previous) => ({ ...previous, [point.timestamp]: true }))} disabled={savingFeedback || hasSaved} title="Confirm as a true anomaly">Yes</button>
                  <button className={`feedback-btn feedback-btn--no ${selected === false ? 'feedback-btn--active' : ''}`} onClick={() => setLocalFeedback((previous) => ({ ...previous, [point.timestamp]: false }))} disabled={savingFeedback || hasSaved} title="Mark as a false positive">No</button>
                  {hasSaved && <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>saved</span>}
                </>;
              })()}
            </div></td>
            <td>{point.timestamp}</td>
            <td>{point.temperature}</td>
            <td>{point.model_quality === 'IMPLAUSIBLE' ? 'Physics veto' : point.model_quality === 'STATISTICAL_OUTLIER' ? 'Isolation Forest outlier' : 'Human-confirmed anomaly'}{point.feedback_applied ? ` · ${point.final_anomaly ? 'feedback: anomaly' : 'feedback: false positive'}` : ''}</td>
            <td>{point.isolation_forest_score?.toFixed(5)}</td>
            <td>{point.reason}</td>
          </tr>)}
        </tbody></table></div>}
    </section>
  </div>;
}
