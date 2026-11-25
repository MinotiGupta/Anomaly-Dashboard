import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { submitFeedback } from '../api';

const formatTime = (value) => {
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? value
    : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};
function ChartTooltip({ active, payload, localFeedback, savedFeedback, savingFeedback, onReview, isFrozen = false, hideWhenFrozen = false }) {
  if (hideWhenFrozen) return null;
  if (!active || !payload || !payload.length) return null;
  const point = payload[0].payload;
  const hasLocal = Object.prototype.hasOwnProperty.call(localFeedback, point.timestamp);
  const hasSaved = Object.prototype.hasOwnProperty.call(savedFeedback, point.timestamp);
  const selected = hasLocal ? localFeedback[point.timestamp] : savedFeedback[point.timestamp];

  return (
    <div
      onMouseDown={(e) => e.stopPropagation()}
      onMouseUp={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      style={{
        background: 'var(--bg-card, #ffffff)',
        border: isFrozen ? '1.5px solid #2563eb' : '1px solid var(--border-default, #e2e8f0)',
        borderRadius: 8,
        padding: '10px 12px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.14)',
        fontSize: '0.82rem',
        pointerEvents: 'auto',
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{point.timestamp}</div>
      <div style={{ color: 'var(--text-secondary)' }}>
        Temperature: {Number(point.temperature).toFixed(3)} °C
      </div>
      {point.reason && (
        <div style={{ color: '#d97706', marginTop: 2 }}>{point.reason}</div>
      )}
      <div className="feedback-buttons" style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Anomaly?</span>
        <button
          type="button"
          className={`feedback-btn feedback-btn--yes ${selected === true ? 'feedback-btn--active' : ''}`}
          onClick={(e) => { e.stopPropagation(); onReview(point.timestamp, true); }}
          disabled={savingFeedback || hasSaved}
        >
          Yes
        </button>
        <button
          type="button"
          className={`feedback-btn feedback-btn--no ${selected === false ? 'feedback-btn--active' : ''}`}
          onClick={(e) => { e.stopPropagation(); onReview(point.timestamp, false); }}
          disabled={savingFeedback || hasSaved}
        >
          No
        </button>
        {hasSaved && <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>saved</span>}
      </div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>
        {isFrozen ? 'Pinned · click chart to resume hover' : 'Click point to freeze'}
      </div>
    </div>
  );
}


function TimeSeriesChart({
  title,
  description,
  data,
  legendItems = [],
  localFeedback = {},
  savedFeedback = {},
  savingFeedback = false,
  onReview,
  children,
}) {
  const [zoomRange, setZoomRange] = useState({ start: 0, end: Math.max(0, data.length - 1) });
  const [dragStart, setDragStart] = useState(null);
  const [dragEnd, setDragEnd] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [frozenTooltip, setFrozenTooltip] = useState(null);

  useEffect(() => {
    setZoomRange({ start: 0, end: Math.max(0, data.length - 1) });
    setActiveIndex(0);
    setDragStart(null);
    setDragEnd(null);
    setFrozenTooltip(null);
  }, [data]);

  // Reset zoom and slider when a new CSV/run loads
 

  const visibleData = useMemo(
    () => data.slice(zoomRange.start, zoomRange.end + 1),
    [data, zoomRange]
  );

  const isZoomed = zoomRange.start > 0 || zoomRange.end < data.length - 1;
  const clampedIndex = Math.min(Math.max(activeIndex, zoomRange.start), zoomRange.end);
  const activePoint = data[clampedIndex] || null;

  const hasLocal = activePoint && Object.prototype.hasOwnProperty.call(localFeedback, activePoint.timestamp);
  const hasSaved = activePoint && Object.prototype.hasOwnProperty.call(savedFeedback, activePoint.timestamp);
  const activeSelected = activePoint
    ? (hasLocal ? localFeedback[activePoint.timestamp] : savedFeedback[activePoint.timestamp])
    : undefined;

  const handleMouseDown = (state) => {
    if (state?.activeLabel) {
      setDragStart(state.activeLabel);
      setDragEnd(state.activeLabel);
    }
  };

  const handleMouseMove = (state) => {
    if (dragStart && state?.activeLabel) {
      setDragEnd(state.activeLabel);
    }
  };

const handleMouseUp = (state) => {
    if (!dragStart || !dragEnd) {
      setDragStart(null);
      setDragEnd(null);
      return;
    }

    const idxA = data.findIndex((p) => p.timestamp === dragStart);
    const idxB = data.findIndex((p) => p.timestamp === dragEnd);

    if (idxA !== -1 && idxB !== -1) {
      if (idxA === idxB) {
        // Single click: sync slider and toggle freeze on/off
        setActiveIndex(idxA);
        setFrozenTooltip((prev) =>
          prev
            ? null
            : {
                point: data[idxA],
                coordinate: state?.activeCoordinate || { x: 160, y: 120 },
              }
        );
      } else {
        // Drag selection: zoom into the selected range
        const [left, right] = idxA < idxB ? [idxA, idxB] : [idxB, idxA];
        setZoomRange({ start: left, end: right });
        setActiveIndex(left);
        setFrozenTooltip(null);
      }
    }

    setDragStart(null);
    setDragEnd(null);
  };

  const handleResetZoom = () => {
    setZoomRange({ start: 0, end: Math.max(0, data.length - 1) });
  };

  const handleStep = (delta) => {
    setActiveIndex((prev) => Math.min(Math.max(prev + delta, zoomRange.start), zoomRange.end));
  };

  return <section className="card" style={{ marginBottom: 20 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
      <div>
        <h3 style={{ marginTop: 0 }}>{title}</h3>
        <p style={{ color: 'var(--text-secondary)', marginTop: -8 }}>{description}</p>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap', fontSize: '0.82rem', paddingTop: 4 }}>
        {legendItems.map((item) => (
          <span key={item.value} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: 'var(--text-secondary)' }}>
            {item.type === 'circle' && (
              <span style={{ width: 9, height: 9, borderRadius: '50%', backgroundColor: item.color, display: 'inline-block' }} />
            )}
            {item.type === 'cross' && (
              <span style={{ color: item.color, fontWeight: 700, fontSize: '0.95rem', lineHeight: 1 }}>✕</span>
            )}
            {item.type === 'plainline' && (
              <span style={{ width: 14, height: 2, backgroundColor: item.color, display: 'inline-block' }} />
            )}
            <span>{item.value}</span>
          </span>
        ))}
        {isZoomed && (
          <button
            type="button"
            className="btn btn--ghost"
            style={{ padding: '4px 10px', fontSize: '0.75rem' }}
            onClick={handleResetZoom}
          >
            Reset Zoom ({visibleData.length} / {data.length} pts)
          </button>
        )}
      </div>
    </div>

    <div style={{ width: '100%', height: 380, userSelect: 'none', position: 'relative' }}>
      <ResponsiveContainer>
        <LineChart
          data={visibleData}
          margin={{ top: 8, right: 24, left: 8, bottom: 24 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
          <XAxis
            dataKey="timestamp"
            type="category"
            tickFormatter={formatTime}
            minTickGap={48}
            height={48}
            label={{ value: 'Time', position: 'insideBottom', offset: -10 }}
          />
          <YAxis
            domain={['auto', 'auto']}
            label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            content={
              <ChartTooltip
                localFeedback={localFeedback}
                savedFeedback={savedFeedback}
                savingFeedback={savingFeedback}
                onReview={onReview}
                hideWhenFrozen={Boolean(frozenTooltip)}
              />
            }
          />
          {children}

          {/* Highlight the active slider point on the plot */}
          {activePoint && (
            <ReferenceLine
              x={activePoint.timestamp}
              stroke="#0f172a"
              strokeDasharray="4 4"
              strokeWidth={1.5}
            />
          )}
          {activePoint && (
            <ReferenceDot
              x={activePoint.timestamp}
              y={activePoint.temperature}
              r={6}
              fill="#0f172a"
              stroke="#ffffff"
              strokeWidth={2}
            />
          )}

          {/* Shaded selection box while dragging to zoom */}
          {dragStart && dragEnd && dragStart !== dragEnd && (
            <ReferenceArea
              x1={dragStart}
              x2={dragEnd}
              strokeOpacity={0.3}
              fill="#2563eb"
              fillOpacity={0.15}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      {frozenTooltip && (
        <div
          style={{
            position: 'absolute',
            left: frozenTooltip.coordinate.x,
            top: Math.max(120, frozenTooltip.coordinate.y),
            transform: 'translate(-50%, -105%)',
            zIndex: 20,
          }}
        >
          <ChartTooltip
            active
            payload={[{ payload: frozenTooltip.point }]}
            localFeedback={localFeedback}
            savedFeedback={savedFeedback}
            savingFeedback={savingFeedback}
            onReview={onReview}
            isFrozen
          />
          </div>
      )}
      
    </div>

    {/* Sequential point slider + Yes/No review controls */}
    {activePoint && (
      <div style={{
        marginTop: 12,
        padding: '10px 14px',
        borderRadius: 8,
        border: '1px solid var(--border-default)',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        fontSize: '0.82rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            type="button"
            className="btn btn--ghost"
            style={{ padding: '4px 10px', fontSize: '0.78rem' }}
            onClick={() => handleStep(-1)}
            disabled={clampedIndex <= zoomRange.start}
          >
            ◀ Prev
          </button>
          <input
            type="range"
            min={zoomRange.start}
            max={zoomRange.end}
            value={clampedIndex}
            onChange={(e) => setActiveIndex(Number(e.target.value))}
            style={{ flex: 1, cursor: 'pointer' }}
          />
          <button
            type="button"
            className="btn btn--ghost"
            style={{ padding: '4px 10px', fontSize: '0.78rem' }}
            onClick={() => handleStep(1)}
            disabled={clampedIndex >= zoomRange.end}
          >
            Next ▶
          </button>
          <span style={{ color: 'var(--text-muted)', minWidth: 90, textAlign: 'right' }}>
            Point {clampedIndex + 1} / {data.length}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <strong>{activePoint.timestamp}</strong> · {Number(activePoint.temperature).toFixed(3)} °C
            {activePoint.reason && (
              <span style={{ color: '#d97706', marginLeft: 8 }}>({activePoint.reason})</span>
            )}
          </div>
          <div className="feedback-buttons" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: 'var(--text-muted)' }}>Mark as anomaly:</span>
            <button
              type="button"
              className={`feedback-btn feedback-btn--yes ${activeSelected === true ? 'feedback-btn--active' : ''}`}
              onClick={() => onReview(activePoint.timestamp, true)}
              disabled={savingFeedback || hasSaved}
            >
              Yes
            </button>
            <button
              type="button"
              className={`feedback-btn feedback-btn--no ${activeSelected === false ? 'feedback-btn--active' : ''}`}
              onClick={() => onReview(activePoint.timestamp, false)}
              disabled={savingFeedback || hasSaved}
            >
              No
            </button>
            {hasSaved && <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>saved</span>}
          </div>
        </div>
      </div>
    )}
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
      const point = data.points.find((reading) => reading.timestamp === timestamp);
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
      legendItems={[
        { value: 'Valid (obeys physics)', type: 'circle', color: '#2563eb' },
        { value: 'IMPLAUSIBLE (vetoed)', type: 'cross', color: '#dc2626' },
        { value: 'Raw trajectory', type: 'plainline', color: '#d1d5db' },
      ]}
      localFeedback={localFeedback}
      savedFeedback={savedFeedback}
      savingFeedback={savingFeedback}
      onReview={(timestamp, isTrueAnomaly) =>
        setLocalFeedback((previous) => ({ ...previous, [timestamp]: isTrueAnomaly }))
      }
    >
      <Line dataKey="temperature" name="Raw trajectory" stroke="#d1d5db" strokeWidth={1.5} dot={false} isAnimationActive={false} />
      <Line dataKey="physicsValidTemperature" name="Valid (obeys physics)" stroke="none" dot={{ r: 2.5, fill: '#2563eb', stroke: '#2563eb' }} activeDot={false} isAnimationActive={false} />
      <Line dataKey="physicsTemperature" name="IMPLAUSIBLE (vetoed)" stroke="none" dot={<PhysicsVetoDot />} activeDot={false} isAnimationActive={false} />
    </TimeSeriesChart>

    <TimeSeriesChart
      title="Hybrid anomaly detection: Physics + Isolation Forest"
      description="Physics vetoes take precedence; Isolation Forest outliers are shown among physics-valid readings."
      data={chartData}
      legendItems={[
        { value: 'Normal operation', type: 'circle', color: '#2563eb' },
        { value: 'Isolation Forest outlier', type: 'circle', color: '#f08c00' },
        { value: 'Physics veto (impossible)', type: 'cross', color: '#dc2626' },
        { value: 'Raw trajectory', type: 'plainline', color: '#d1d5db' },
      ]}
      localFeedback={localFeedback}
      savedFeedback={savedFeedback}
      savingFeedback={savingFeedback}
      onReview={(timestamp, isTrueAnomaly) =>
        setLocalFeedback((previous) => ({ ...previous, [timestamp]: isTrueAnomaly }))
      }
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
