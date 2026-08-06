import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export default function ConfidenceDistributionChart({ data, height = 280 }) {
  if (!data || !data.counts || data.counts.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height, color: 'var(--text-muted)', fontSize: '0.82rem' }}>
        No anomalies detected — no confidence data available.
      </div>
    );
  }

  const chartData = data.counts.map((count, i) => {
    const binStart = data.bin_edges[i];
    const binEnd = data.bin_edges[i + 1];
    const binCenter = (binStart + binEnd) / 2;
    return {
      bin: `${binStart.toFixed(0)}-${binEnd.toFixed(0)}%`,
      binCenter,
      count,
    };
  });

  const getBarColor = (center) => {
    if (center >= 85) return 'rgba(239, 68, 68, 0.7)';
    if (center >= 70) return 'rgba(245, 158, 11, 0.6)';
    return 'rgba(59, 130, 246, 0.5)';
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.06)" vertical={false} />
        <XAxis
          dataKey="bin"
          tick={{ fill: '#64748b', fontSize: 9 }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          interval={1}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          width={35}
          allowDecimals={false}
        />
        <Tooltip
          contentStyle={{
            background: '#243047',
            border: '1px solid rgba(148,163,184,0.12)',
            borderRadius: '8px',
            fontSize: '0.78rem',
          }}
          labelStyle={{ color: '#f1f5f9', fontWeight: 500 }}
          formatter={(value) => [value, 'Anomalies']}
        />
        <Bar dataKey="count" radius={[3, 3, 0, 0]} maxBarSize={28}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={getBarColor(entry.binCenter)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
