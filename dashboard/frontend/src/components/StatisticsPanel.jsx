import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ErrorBar,
  ComposedChart,
  Scatter,
  Line,
  ReferenceLine,
} from 'recharts';

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#22c55e', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6'];

export default function StatisticsPanel({ channelStats }) {
  if (!channelStats) return null;

  const channels = Object.keys(channelStats);
  if (channels.length === 0) return null;

  // Skewness & Kurtosis data for the bar chart
  const skewKurtData = channels.map((ch) => ({
    channel: ch,
    skewness: parseFloat(channelStats[ch].skewness?.toFixed(4)) || 0,
    kurtosis: parseFloat(channelStats[ch].kurtosis?.toFixed(4)) || 0,
  }));

  // Confidence Interval data
  const ciData = channels.map((ch) => ({
    channel: ch,
    mean: channelStats[ch].mean,
    ci_low: channelStats[ch].ci_95_low,
    ci_high: channelStats[ch].ci_95_high,
    error: channelStats[ch].ci_95_high - channelStats[ch].mean,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Summary Statistics Table */}
      <div className="card">
        <div className="card__header">
          <span className="card__title">Per-Channel Statistics</span>
          <span className="card__badge card__badge--blue">{channels.length} channels</span>
        </div>
        <div className="anomaly-table-wrapper">
          <table className="anomaly-table">
            <thead>
              <tr>
                <th>Channel</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Skewness</th>
                <th>Kurtosis</th>
                <th>95% CI</th>
              </tr>
            </thead>
            <tbody>
              {channels.map((ch) => {
                const s = channelStats[ch];
                return (
                  <tr key={ch}>
                    <td style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{ch}</td>
                    <td>{s.mean?.toFixed(4)}</td>
                    <td>{s.std?.toFixed(4)}</td>
                    <td>{s.min?.toFixed(4)}</td>
                    <td>{s.max?.toFixed(4)}</td>
                    <td style={{ color: getSkewColor(s.skewness) }}>{s.skewness?.toFixed(4)}</td>
                    <td style={{ color: getKurtColor(s.kurtosis) }}>{s.kurtosis?.toFixed(4)}</td>
                    <td style={{ fontSize: '0.72rem' }}>
                      [{s.ci_95_low?.toFixed(3)}, {s.ci_95_high?.toFixed(3)}]
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts row */}
      <div className="chart-grid">
        {/* Skewness Chart */}
        <div className="card">
          <div className="card__header">
            <span className="card__title">Skewness by Channel</span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={skewKurtData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
              <XAxis
                dataKey="channel"
                tick={{ fill: '#94a3b8', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                tickLine={false}
                interval={0}
                angle={channels.length > 6 ? -30 : 0}
                textAnchor={channels.length > 6 ? 'end' : 'middle'}
                height={channels.length > 6 ? 60 : 30}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                tickLine={false}
                width={50}
              />
              <Tooltip
                contentStyle={{
                  background: '#243047',
                  border: '1px solid rgba(148,163,184,0.12)',
                  borderRadius: '8px',
                  fontSize: '0.78rem',
                }}
                labelStyle={{ color: '#f1f5f9', fontWeight: 500 }}
                formatter={(value) => [value.toFixed(4), 'Skewness']}
              />
              <ReferenceLine y={0} stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
              <Bar dataKey="skewness" radius={[3, 3, 0, 0]} maxBarSize={32}>
                {skewKurtData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.skewness >= 0
                      ? 'rgba(59, 130, 246, 0.6)'
                      : 'rgba(245, 158, 11, 0.6)'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4, padding: '0 4px' }}>
            Positive skew = right-tailed. Negative skew = left-tailed. Near 0 = symmetric.
          </p>
        </div>

        {/* Kurtosis Chart */}
        <div className="card">
          <div className="card__header">
            <span className="card__title">Kurtosis by Channel</span>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={skewKurtData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
              <XAxis
                dataKey="channel"
                tick={{ fill: '#94a3b8', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                tickLine={false}
                interval={0}
                angle={channels.length > 6 ? -30 : 0}
                textAnchor={channels.length > 6 ? 'end' : 'middle'}
                height={channels.length > 6 ? 60 : 30}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                tickLine={false}
                width={50}
              />
              <Tooltip
                contentStyle={{
                  background: '#243047',
                  border: '1px solid rgba(148,163,184,0.12)',
                  borderRadius: '8px',
                  fontSize: '0.78rem',
                }}
                labelStyle={{ color: '#f1f5f9', fontWeight: 500 }}
                formatter={(value) => [value.toFixed(4), 'Kurtosis']}
              />
              <ReferenceLine y={0} stroke="rgba(148,163,184,0.2)" strokeDasharray="3 3" />
              <Bar dataKey="kurtosis" radius={[3, 3, 0, 0]} maxBarSize={32}>
                {skewKurtData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.kurtosis > 0
                      ? 'rgba(139, 92, 246, 0.6)'
                      : 'rgba(6, 182, 212, 0.6)'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4, padding: '0 4px' }}>
            Positive kurtosis = heavy tails (leptokurtic). Negative = light tails (platykurtic).
          </p>
        </div>
      </div>

      {/* 95% Confidence Interval Chart */}
      <div className="card">
        <div className="card__header">
          <span className="card__title">95% Confidence Interval for Channel Means</span>
        </div>
        <ResponsiveContainer width="100%" height={Math.max(200, channels.length * 42 + 40)}>
          <BarChart
            data={ciData}
            layout="vertical"
            margin={{ top: 8, right: 30, bottom: 4, left: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
              tickLine={false}
              tickFormatter={(v) => v.toFixed(1)}
            />
            <YAxis
              dataKey="channel"
              type="category"
              tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
              tickLine={false}
              width={100}
            />
            <Tooltip
              contentStyle={{
                background: '#243047',
                border: '1px solid rgba(148,163,184,0.12)',
                borderRadius: '8px',
                fontSize: '0.78rem',
              }}
              labelStyle={{ color: '#f1f5f9', fontWeight: 500 }}
              formatter={(value, name, props) => {
                const d = props.payload;
                return [
                  `${d.mean.toFixed(4)} [${d.ci_low.toFixed(4)}, ${d.ci_high.toFixed(4)}]`,
                  'Mean [95% CI]',
                ];
              }}
            />
            <Bar dataKey="mean" radius={[0, 4, 4, 0]} maxBarSize={20} fillOpacity={0.7}>
              {ciData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
              <ErrorBar dataKey="error" width={8} stroke="#94a3b8" strokeWidth={1.5} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4, padding: '0 4px' }}>
          Error bars show the 95% confidence interval for each channel mean.
        </p>
      </div>
    </div>
  );
}

function getSkewColor(skew) {
  if (skew == null) return 'var(--text-secondary)';
  const abs = Math.abs(skew);
  if (abs > 1) return '#ef4444';
  if (abs > 0.5) return '#f59e0b';
  return '#22c55e';
}

function getKurtColor(kurt) {
  if (kurt == null) return 'var(--text-secondary)';
  const abs = Math.abs(kurt);
  if (abs > 3) return '#ef4444';
  if (abs > 1) return '#f59e0b';
  return '#22c55e';
}
