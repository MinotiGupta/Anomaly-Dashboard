export default function MetadataPanel({ metadata }) {
  if (!metadata) return null;

  const fields = [
    { key: 'model', label: 'Instrument Model' },
    { key: 'serial_number', label: 'Serial Number' },
    { key: 'firmware_version', label: 'Firmware Version' },
    { key: 'address', label: 'Address' },
    { key: 'start_time', label: 'Recording Start' },
    { key: 'stop_time', label: 'Recording Stop' },
    { key: 'total_channels', label: 'Total Channels' },
    { key: 'modules', label: 'Modules' },
  ];

  const hasChannelConfig =
    metadata.channel_config && metadata.channel_config.length > 0;

  return (
    <div className="card fade-in" style={{ marginBottom: '20px' }}>
      <div className="card__header">
        <span className="card__title">Instrument Metadata</span>
        <span className="card__badge card__badge--blue">
          {metadata.model || 'Hardware'}
        </span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap: '12px',
          marginBottom: hasChannelConfig ? '16px' : 0,
        }}
      >
        {fields.map(({ key, label }) => {
          const value = metadata[key];
          if (!value) return null;
          return (
            <div key={key}>
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
                  fontSize: '0.85rem',
                  color: 'var(--text-secondary)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {value}
              </span>
            </div>
          );
        })}
      </div>

      {hasChannelConfig && (
        <div>
          <p
            style={{
              fontSize: '0.75rem',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--text-muted)',
              marginBottom: 8,
              paddingTop: 12,
              borderTop: '1px solid var(--border-subtle)',
            }}
          >
            Channel Configuration
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {metadata.channel_config.map((ch, i) => (
              <div
                key={i}
                style={{
                  padding: '4px 10px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-elevated)',
                  fontSize: '0.75rem',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                }}
              >
                {ch.channel_id}
                {ch.function ? ` / ${ch.function}` : ''}
                {ch.unit ? ` (${ch.unit})` : ''}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
