import { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export default function ChannelChart({ channelName, channelData, height = 300 }) {
  const chartData = useMemo(() => {
    const anomalySet = new Set(channelData.anomaly_timestamps);
    return channelData.timestamps.map((ts, i) => {
      const isAnomaly = anomalySet.has(ts);
      const anomalyIdx = isAnomaly
        ? channelData.anomaly_timestamps.indexOf(ts)
        : -1;

      return {
        timestamp: ts,
        value: channelData.values[i],
        anomaly: isAnomaly ? channelData.anomaly_values[anomalyIdx] : null,
        confidence: isAnomaly
          ? channelData.anomaly_confidences[anomalyIdx]
          : null,
      };
    });
  }, [channelData]);

  // Downsample for rendering performance if > 2000 points
  const displayData = useMemo(() => {
    if (chartData.length <= 2000) return chartData;
    const step = Math.ceil(chartData.length / 2000);
    const sampled = [];
    for (let i = 0; i < chartData.length; i++) {
      if (i % step === 0 || chartData[i].anomaly !== null) {
        sampled.push(chartData[i]);
      }
    }
    return sampled;
  }, [chartData]);

  const formatTimestamp = (ts) => {
    if (!ts) return '';
    const str = String(ts);
    // Extract time portion if it looks like a datetime
    const match = str.match(/(\d{2}:\d{2}:\d{2})/);
    return match ? match[1] : str.slice(-8);
  };

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null;
    const d = payload[0]?.payload;
    if (!d) return null;

    return (
      <div
        style={{
          background: '#243047',
          border: '1px solid rgba(148,163,184,0.12)',
          borderRadius: '8px',
          padding: '10px 14px',
          fontSize: '0.78rem',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
        }}
      >
        <div style={{ color: '#f1f5f9', fontWeight: 500, marginBottom: 4 }}>
          {d.timestamp}
        </div>
        <div style={{ color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
          Value: {d.value?.toFixed(3)}
        </div>
        {d.anomaly !== null && (
          <div style={{ color: '#ef4444', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
            Anomaly — Confidence: {d.confidence?.toFixed(1)}%
          </div>
        )}
      </div>
    );
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={displayData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="rgba(148,163,184,0.06)"
          vertical={false}
        />
        <XAxis
          dataKey="timestamp"
          tickFormatter={formatTimestamp}
          tick={{ fill: '#64748b', fontSize: 10 }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          interval="preserveStartEnd"
          minTickGap={60}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
          tickLine={false}
          width={55}
        />
        <Tooltip content={<CustomTooltip />} />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          strokeWidth={1.5}
          dot={false}
          activeDot={false}
          isAnimationActive={false}
        />
        <Scatter
          dataKey="anomaly"
          fill="#ef4444"
          stroke="#991b1b"
          strokeWidth={1}
          r={4}
          shape="diamond"
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
