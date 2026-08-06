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

const COLORS = ['#3b82f6', '#8b5cf6', '#06b6d4', '#22c55e', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6'];

export default function FeatureContributionChart({ data, height = 300 }) {
  const chartData = data.features.map((feat, i) => ({
    feature: feat,
    mae: data.mae_values[i],
  }));

  // Sort descending by mae
  chartData.sort((a, b) => b.mae - a.mae);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 8, right: 20, bottom: 4, left: 10 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(148,163,184,0.06)"
          horizontal={false}
        />
        <XAxis
          type="number"
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          tickFormatter={(v) => v.toFixed(3)}
        />
        <YAxis
          dataKey="feature"
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
          formatter={(value) => [value.toFixed(6), 'MAE']}
        />
        <Bar dataKey="mae" radius={[0, 4, 4, 0]} maxBarSize={24}>
          {chartData.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
