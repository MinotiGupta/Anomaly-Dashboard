import { useState } from 'react';
import ChannelChart from './ChannelChart';
import TrainingLossChart from './TrainingLossChart';
import MAEDistributionChart from './MAEDistributionChart';
import FeatureContributionChart from './FeatureContributionChart';
import ConfidenceDistributionChart from './ConfidenceDistributionChart';
import AnomalyTable from './AnomalyTable';
import MetadataPanel from './MetadataPanel';
import StatisticsPanel from './StatisticsPanel';
import KDEChart from './KDEChart';

const DATA_TYPE_LABELS = {
  biomass: 'Biomass',
  cataluminescence: 'Cataluminescence',
  swiss_roll: 'Swiss Roll',
  default: 'Default (34970A)',
  unknown: 'Unknown',
};

export default function ResultsDashboard({ data, runId, onRerun }) {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'channels', label: 'Sensor Channels' },
    { id: 'kde', label: 'KDE Threshold' },
    { id: 'diagnostics', label: 'Model Diagnostics' },
    { id: 'table', label: 'Anomaly Log' },
    { id: 'statistics', label: 'Statistics' },
  ];

  return (
    <div className="fade-in">
      {/* Data type + feedback banner */}
      <div
        className="card"
        style={{
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Classified As
            </span>
            <p style={{ fontSize: '1rem', fontWeight: 600, marginTop: 2 }}>
              {DATA_TYPE_LABELS[data.data_type] || data.data_type}
            </p>
          </div>
          {data.used_feedback && (
            <span className="card__badge card__badge--green">
              Trained with feedback
            </span>
          )}
          {data.file_name && (
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                File
              </span>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
                {data.file_name}
              </p>
            </div>
          )}
        </div>
        <button className="btn btn--ghost" onClick={onRerun} style={{ fontSize: '0.8rem' }}>
          Retrain with feedback
        </button>
      </div>

      {/* Instrument Metadata (if available) */}
      {data.instrument_metadata && (
        <MetadataPanel metadata={data.instrument_metadata} />
      )}

      {/* Summary Stats */}
      <div className="stats-row fade-in fade-in-delay-1">
        <div className="card stat-card">
          <p className="stat-card__label">Data Points</p>
          <p className="stat-card__value stat-card__value--blue">
            {data.total_points.toLocaleString()}
          </p>
        </div>
        <div className="card stat-card">
          <p className="stat-card__label">Sensor Channels</p>
          <p className="stat-card__value stat-card__value--green">
            {data.summary.total_channels}
          </p>
        </div>
        <div className="card stat-card">
          <p className="stat-card__label">Anomalies Detected</p>
          <p className="stat-card__value stat-card__value--red">
            {data.total_anomalies}
          </p>
        </div>
        <div className="card stat-card">
          <p className="stat-card__label">Anomaly Rate</p>
          <p className="stat-card__value stat-card__value--amber">
            {data.anomaly_rate}%
          </p>
        </div>
      </div>

      {/* Tab Bar */}
      <div className="tab-bar fade-in fade-in-delay-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`tab-bar__item ${activeTab === tab.id ? 'tab-bar__item--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="fade-in fade-in-delay-3">
        {activeTab === 'overview' && <OverviewTab data={data} />}
        {activeTab === 'channels' && <ChannelsTab data={data} />}
        {activeTab === 'kde' && <KDETab data={data} />}
        {activeTab === 'diagnostics' && <DiagnosticsTab data={data} />}
        {activeTab === 'table' && (
          <AnomalyTable
            anomalies={data.anomaly_table}
            sensorColumns={data.summary.sensor_columns}
            dataType={data.data_type}
            fileName={data.file_name}
            runId={runId}
            existingFeedback={data.existing_feedback || {}}
          />
        )}
        {activeTab === 'statistics' && (
          <StatisticsPanel channelStats={data.channel_statistics} />
        )}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  const channelNames = Object.keys(data.channels);

  return (
    <>
      {/* All channel charts (one per channel) */}
      {channelNames.map((ch) => (
        <div className="card" key={ch} style={{ marginBottom: '20px' }}>
          <div className="card__header">
            <span className="card__title">{ch}</span>
            <span className="card__badge card__badge--red">
              {data.channels[ch].anomaly_timestamps.length} anomalies
            </span>
          </div>
          <ChannelChart
            channelName={ch}
            channelData={data.channels[ch]}
            height={240}
          />
        </div>
      ))}

      {/* Training loss + Confidence distribution */}
      <div className="chart-grid">
        <div className="card">
          <div className="card__header">
            <span className="card__title">Training Loss (MAE)</span>
            <span className="card__badge card__badge--green">30 epochs</span>
          </div>
          <TrainingLossChart data={data.training_loss} height={280} />
        </div>

        <div className="card">
          <div className="card__header">
            <span className="card__title">Anomaly Confidence Distribution</span>
            <span className="card__badge card__badge--amber">
              {data.total_anomalies} anomalies
            </span>
          </div>
          <ConfidenceDistributionChart
            data={data.confidence_distribution}
            height={280}
          />
        </div>
      </div>

      {/* MAE Distribution + Feature Contribution */}
      <div className="chart-grid">
        <div className="card">
          <div className="card__header">
            <span className="card__title">Reconstruction Error Distribution</span>
            <span className="card__badge card__badge--amber">
              threshold: {data.mae_distribution.threshold.toFixed(4)}
            </span>
          </div>
          <MAEDistributionChart data={data.mae_distribution} height={280} />
        </div>

        <div className="card">
          <div className="card__header">
            <span className="card__title">Per-Feature Error Contribution</span>
          </div>
          <FeatureContributionChart data={data.feature_contribution} height={280} />
        </div>
      </div>
    </>
  );
}

function ChannelsTab({ data }) {
  const channelNames = Object.keys(data.channels);
  const [selectedChannel, setSelectedChannel] = useState(channelNames[0]);

  return (
    <>
      <div className="channel-pills">
        {channelNames.map((name) => (
          <button
            key={name}
            className={`channel-pill ${selectedChannel === name ? 'channel-pill--active' : ''}`}
            onClick={() => setSelectedChannel(name)}
          >
            {name}
          </button>
        ))}
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card__header">
          <span className="card__title">{selectedChannel}</span>
          <span className="card__badge card__badge--red">
            {data.channels[selectedChannel].anomaly_timestamps.length} anomalies
          </span>
        </div>
        <ChannelChart
          channelName={selectedChannel}
          channelData={data.channels[selectedChannel]}
          height={360}
        />
      </div>

      {/* Per-channel stats for selected channel */}
      {data.channel_statistics && data.channel_statistics[selectedChannel] && (
        <div className="card">
          <div className="card__header">
            <span className="card__title">Channel Statistics: {selectedChannel}</span>
          </div>
          <ChannelStatsGrid stats={data.channel_statistics[selectedChannel]} />
        </div>
      )}
    </>
  );
}

function ChannelStatsGrid({ stats }) {
  const items = [
    { label: 'Mean', value: stats.mean?.toFixed(4) },
    { label: 'Std Dev', value: stats.std?.toFixed(4) },
    { label: 'Min', value: stats.min?.toFixed(4) },
    { label: 'Max', value: stats.max?.toFixed(4) },
    { label: 'Skewness', value: stats.skewness?.toFixed(4), color: getSkewColor(stats.skewness) },
    { label: 'Kurtosis', value: stats.kurtosis?.toFixed(4), color: getKurtColor(stats.kurtosis) },
    { label: '95% CI Low', value: stats.ci_95_low?.toFixed(4) },
    { label: '95% CI High', value: stats.ci_95_high?.toFixed(4) },
  ];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
        gap: '12px',
      }}
    >
      {items.map(({ label, value, color }) => (
        <div key={label}>
          <span
            style={{
              fontSize: '0.7rem',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--text-muted)',
              display: 'block',
              marginBottom: 2,
            }}
          >
            {label}
          </span>
          <span
            style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: color || 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

function KDETab({ data }) {
  return (
    <div className="chart-grid chart-grid--full">
      <div className="card">
        <div className="card__header">
          <span className="card__title">Kernel Density Estimation — Reconstruction Error</span>
          {data.kde_data && (
            <span className="card__badge card__badge--red">
              threshold: {Number(data.kde_data.threshold).toFixed(4)}
            </span>
          )}
        </div>
        <KDEChart kdeData={data.kde_data} height={380} />
      </div>

      {/* Side-by-side: MAE histogram + confidence distribution */}
      <div className="chart-grid">
        <div className="card">
          <div className="card__header">
            <span className="card__title">MAE Score Distribution (Histogram)</span>
          </div>
          <MAEDistributionChart data={data.mae_distribution} height={280} />
        </div>
        <div className="card">
          <div className="card__header">
            <span className="card__title">Anomaly Confidence Distribution</span>
          </div>
          <ConfidenceDistributionChart data={data.confidence_distribution} height={280} />
        </div>
      </div>

      {/* Threshold info table */}
      {data.kde_data && (
        <div className="card">
          <div className="card__header">
            <span className="card__title">Threshold Details</span>
          </div>
          <div style={{ padding: '8px 0' }}>
            <ConfigRow label="Active Threshold" value={Number(data.kde_data.threshold).toFixed(6)} />
            <ConfigRow label="Decision Method" value={data.kde_data.threshold_method} />
            <ConfigRow label="KDE Natural Valley" value={Number(data.kde_data.kde_threshold).toFixed(6)} />
            <ConfigRow label="KDE Detection Method" value={data.kde_data.kde_method} />
            <ConfigRow label="Total Anomalies" value={data.total_anomalies} />
            <ConfigRow label="Anomaly Rate" value={`${data.anomaly_rate}%`} />
          </div>
        </div>
      )}
    </div>
  );
}

function DiagnosticsTab({ data }) {
  return (
    <div className="chart-grid">
      <div className="card">
        <div className="card__header">
          <span className="card__title">Training Loss Curve</span>
          <span className="card__badge card__badge--blue">L1 Loss</span>
        </div>
        <TrainingLossChart data={data.training_loss} height={320} />
      </div>

      <div className="card">
        <div className="card__header">
          <span className="card__title">MAE Score Distribution</span>
          <span className="card__badge card__badge--amber">
            threshold: {data.mae_distribution.threshold.toFixed(4)}
          </span>
        </div>
        <MAEDistributionChart data={data.mae_distribution} height={320} />
      </div>

      <div className="card">
        <div className="card__header">
          <span className="card__title">Anomaly Confidence Distribution</span>
        </div>
        <ConfidenceDistributionChart
          data={data.confidence_distribution}
          height={320}
        />
      </div>

      <div className="card">
        <div className="card__header">
          <span className="card__title">Per-Feature Reconstruction Error</span>
        </div>
        <FeatureContributionChart data={data.feature_contribution} height={320} />
      </div>

      <div className="card">
        <div className="card__header">
          <span className="card__title">Model Configuration</span>
        </div>
        <div style={{ padding: '8px 0' }}>
          <ConfigRow label="Architecture" value="LSTM Autoencoder" />
          <ConfigRow label="Input Channels" value={data.summary.total_channels} />
          <ConfigRow label="Sequence Length" value="10 timesteps" />
          <ConfigRow
            label="Hidden Dimension"
            value={`${Math.min(64, Math.max(16, data.summary.total_channels * 4))}`}
          />
          <ConfigRow label="Epochs" value="30" />
          <ConfigRow label="Batch Size" value="16" />
          <ConfigRow label="Loss Function" value="L1 (MAE)" />
          <ConfigRow label="Optimizer" value="Adam (lr=0.001)" />
          <ConfigRow label="Contamination" value="3%" />
          {data.used_feedback && (
            <ConfigRow label="Feedback" value="Enabled (weighted retraining)" />
          )}
        </div>
      </div>
    </div>
  );
}

function ConfigRow({ label, value }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '8px 0',
        borderBottom: '1px solid var(--border-subtle)',
        fontSize: '0.82rem',
      }}
    >
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
        {value}
      </span>
    </div>
  );
}

function getSkewColor(skew) {
  if (skew == null) return undefined;
  const abs = Math.abs(skew);
  if (abs > 1) return '#ef4444';
  if (abs > 0.5) return '#f59e0b';
  return '#22c55e';
}

function getKurtColor(kurt) {
  if (kurt == null) return undefined;
  const abs = Math.abs(kurt);
  if (abs > 3) return '#ef4444';
  if (abs > 1) return '#f59e0b';
  return '#22c55e';
}
