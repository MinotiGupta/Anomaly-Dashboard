import { useMemo } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export default function PhysicsResults({ data }) {
  const chartData = useMemo(() => data.points.map((point, index) => ({
    ...point,
    index: index + 1,
    validTemperature: point.quality === 'VALID' ? point.temperature : null,
    anomalyTemperature: point.quality === 'IMPLAUSIBLE' ? point.temperature : null,
  })), [data.points]);
  const anomalies = data.points.filter((point) => point.quality === 'IMPLAUSIBLE');

  return <div className="fade-in">
    <section className="card" style={{ marginBottom: 20 }}>
      <h2 style={{ marginTop: 0 }}>Physics plausibility analysis</h2>
      <p style={{ color: 'var(--text-secondary)' }}>Temperature channel: {data.channel} · {data.file_name}</p>
      <div className="stats-row">
        <div className="card stat-card"><p className="stat-card__label">Points processed</p><p className="stat-card__value">{data.total_points.toLocaleString()}</p></div>
        <div className="card stat-card"><p className="stat-card__label">IMPLAUSIBLE</p><p className="stat-card__value" style={{ color: 'var(--accent-red)' }}>{data.total_anomalies.toLocaleString()}</p></div>
        <div className="card stat-card"><p className="stat-card__label">Anomaly rate</p><p className="stat-card__value">{data.anomaly_rate}%</p></div>
      </div>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>Limits: {data.config.sensor_min_c}–{data.config.sensor_max_c} °C sensor range, {data.config.sink_c}–{data.config.drive_max_c} °C environment, τ={data.config.tau_seconds}s, margin={data.config.tau_uncertainty_margin}.</p>
    </section>
    <section className="card" style={{ marginBottom: 20 }}>
      <h3>Temperature trajectory</h3>
      <div style={{ width: '100%', height: 360 }}><ResponsiveContainer><LineChart data={chartData} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-default)" />
        <XAxis dataKey="index" label={{ value: 'Sample', position: 'insideBottom', offset: -4 }} />
        <YAxis label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }} />
        <Tooltip labelFormatter={(value) => `Sample ${value}`} formatter={(value, name) => [`${value} °C`, name]} />
        <Legend /><Line dataKey="temperature" name="Raw trajectory" stroke="#9ca3af" dot={false} isAnimationActive={false} connectNulls />
        <Line dataKey="validTemperature" name="Valid" stroke="#2563eb" dot={{ r: 2 }} connectNulls={false} isAnimationActive={false} />
        <Line dataKey="anomalyTemperature" name="IMPLAUSIBLE" stroke="#ef4444" dot={{ r: 4, fill: '#ef4444' }} connectNulls={false} isAnimationActive={false} />
      </LineChart></ResponsiveContainer></div>
    </section>
    <section className="card"><h3>IMPLAUSIBLE readings ({anomalies.length})</h3>
      {anomalies.length === 0 ? <p style={{ color: 'var(--text-muted)' }}>No implausible readings detected.</p> :
        <div className="anomaly-table-wrapper"><table className="anomaly-table"><thead><tr><th>Timestamp</th><th>Temperature (°C)</th><th>Reason</th></tr></thead><tbody>
          {anomalies.map((point, index) => <tr key={`${point.timestamp}-${index}`}><td>{point.timestamp}</td><td>{point.temperature}</td><td>{point.reason}</td></tr>)}
        </tbody></table></div>}
    </section>
  </div>;
}
