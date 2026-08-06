import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export default function MAEDistributionChart({ data, height = 300 }) {
  const chartData = data.counts.map((count, i) => {
    const binStart = data.bin_edges[i];
    const binEnd = data.bin_edges[i + 1];
    const binCenter = (binStart + binEnd) / 2;
    return {
      bin: binCenter.toFixed(4),
      binStart: binStart.toFixed(4),
      binEnd: binEnd.toFixed(4),
      count,
      aboveThreshold: binCenter > data.threshold,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(148,163,184,0.06)"
          vertical={false}
        />
        <XAxis
          dataKey="bin"
          tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          interval="preserveStartEnd"
          minTickGap={40}
          label={{
            value: 'MAE Score',
            position: 'insideBottom',
            offset: -2,
            style: { fill: '#64748b', fontSize: 11 },
          }}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          width={45}
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
            return [value, `Count [${d.binStart} – ${d.binEnd}]`];
          }}
          labelFormatter={() => ''}
        />
        <ReferenceLine
          x={data.threshold.toFixed(4)}
          stroke="#ef4444"
          strokeDasharray="4 4"
          strokeWidth={1.5}
          label={{
            value: 'Threshold',
            position: 'top',
            style: { fill: '#ef4444', fontSize: 10, fontWeight: 500 },
          }}
        />
        <Bar dataKey="count" radius={[2, 2, 0, 0]} maxBarSize={20}>
          {chartData.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.aboveThreshold ? 'rgba(239,68,68,0.6)' : 'rgba(59,130,246,0.5)'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
