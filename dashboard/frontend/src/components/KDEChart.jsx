import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="recharts-default-tooltip">
        <p className="recharts-tooltip-label">MAE: {Number(label).toFixed(6)}</p>
        {payload.map((entry) => (
          <p key={entry.name} className="recharts-tooltip-item" style={{ color: entry.color }}>
            {entry.name}: {Number(entry.value).toFixed(6)}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function KDEChart({ kdeData, height = 320 }) {
  if (!kdeData || !kdeData.x || kdeData.x.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No KDE data available</span>
      </div>
    );
  }

  // Subsample points to max 400 for performance
  const step = Math.max(1, Math.floor(kdeData.x.length / 400));
  const chartData = kdeData.x
    .filter((_, i) => i % step === 0)
    .map((x, i) => ({
      x: Number(x),
      density: Number(kdeData.y[i * step] ?? 0),
    }));

  const threshold = Number(kdeData.threshold);
  const kdeNatural = Number(kdeData.kde_threshold);
  const isLoadedFromModel = kdeData.threshold_method === 'Loaded from saved model';

  return (
    <div>
      {/* Method badge */}
      <div style={{ marginBottom: 12, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Threshold method:</span>
        <span
          className="card__badge card__badge--blue"
          style={{ fontSize: '0.72rem' }}
        >
          {kdeData.threshold_method}
        </span>
        {isLoadedFromModel && (
          <span
            className="card__badge card__badge--green"
            style={{ fontSize: '0.72rem' }}
          >
            KDE natural valley: {kdeNatural.toFixed(6)}
          </span>
        )}
        <span
          className="card__badge card__badge--red"
          style={{ fontSize: '0.72rem' }}
        >
          Decision boundary: {threshold.toFixed(6)}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
          <XAxis
            dataKey="x"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(v) => v.toFixed(4)}
            tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
            label={{ value: 'Reconstruction Error (MAE)', position: 'insideBottom', offset: -2, fontSize: 11, fill: 'var(--text-muted)' }}
          />
          <YAxis
            tickFormatter={(v) => v.toFixed(3)}
            tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
            label={{ value: 'Probability Density', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--text-muted)' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: '0.78rem' }} />

          <Line
            type="monotone"
            dataKey="density"
            name="Error Density"
            stroke="var(--accent-blue)"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
          />

          {/* Active decision threshold (red) */}
          <ReferenceLine
            x={threshold}
            stroke="#ef4444"
            strokeWidth={2}
            strokeDasharray="6 3"
            label={{
              value: `Threshold: ${threshold.toFixed(4)}`,
              position: 'top',
              fontSize: 10,
              fill: '#ef4444',
            }}
          />

          {/* Show KDE natural valley separately when using loaded threshold */}
          {isLoadedFromModel && Math.abs(kdeNatural - threshold) > 1e-6 && (
            <ReferenceLine
              x={kdeNatural}
              stroke="#f59e0b"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              label={{
                value: `KDE valley: ${kdeNatural.toFixed(4)}`,
                position: 'insideTopRight',
                fontSize: 10,
                fill: '#f59e0b',
              }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Explanation text */}
      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 10, lineHeight: 1.5 }}>
        The KDE curve shows the probability density of reconstruction errors across all data points.
        The threshold separates <strong>normal</strong> (left) from <strong>anomalous</strong> (right) behaviour.
        {isLoadedFromModel
          ? ' The decision boundary was computed at training time and persisted for stable, consistent inference.'
          : ' The threshold was found at the first local minimum (valley) of the density curve.'}
      </p>
    </div>
  );
}
