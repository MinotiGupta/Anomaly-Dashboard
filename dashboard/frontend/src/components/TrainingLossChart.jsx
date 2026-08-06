import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export default function TrainingLossChart({ data, height = 300 }) {
  const chartData = data.epochs.map((epoch, i) => ({
    epoch,
    loss: data.losses[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(148,163,184,0.06)"
          vertical={false}
        />
        <XAxis
          dataKey="epoch"
          tick={{ fill: '#64748b', fontSize: 10 }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          label={{
            value: 'Epoch',
            position: 'insideBottom',
            offset: -2,
            style: { fill: '#64748b', fontSize: 11 },
          }}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          width={55}
          tickFormatter={(v) => v.toFixed(3)}
        />
        <Tooltip
          contentStyle={{
            background: '#243047',
            border: '1px solid rgba(148,163,184,0.12)',
            borderRadius: '8px',
            fontSize: '0.78rem',
          }}
          labelStyle={{ color: '#f1f5f9', fontWeight: 500 }}
          itemStyle={{ color: '#22c55e', fontFamily: 'var(--font-mono)' }}
          formatter={(value) => [value.toFixed(6), 'Loss']}
          labelFormatter={(v) => `Epoch ${v}`}
        />
        <Line
          type="monotone"
          dataKey="loss"
          stroke="#22c55e"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, stroke: '#22c55e', fill: '#0a0e17' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
