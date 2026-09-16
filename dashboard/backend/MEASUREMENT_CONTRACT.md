# Measurement Contract v1.0

`measurement_contract.py` defines the canonical representation of one sensor sample.

## Required behavior

- `raw_value` is the value received from the device and is always retained.
- `wall_clock_timestamp` must be timezone-aware and is used for storage and display.
- `monotonic_uptime_seconds` is used for rate and interval calculations.
- `boot_id` allows resets and data gaps to be detected.
- Device, channel, sensor, and front-end identities are explicit.
- Hardware words and register data are preserved when available.
- Detector and parameter versions make results reproducible.
- Process state is carried in `event_metadata`.
- `quality_flags` are additive annotations.

`MeasurementRecord` is immutable. Detector code must call `with_quality_flags()` to create a new record. It must not modify, clamp, smooth, substitute, or delete the raw sample.

## Example

```python
from datetime import datetime, timezone

from measurement_contract import (
    FrontEndType,
    MeasurementRecord,
    QualityClass,
    QualityFlag,
    SensorType,
)

record = MeasurementRecord(
    measurement_id="node-1/boot-7/42",
    raw_value=85.0,
    wall_clock_timestamp=datetime.now(timezone.utc),
    monotonic_uptime_seconds=42.5,
    boot_id="boot-7",
    device_id="node-1",
    channel_id="probe-1",
    sensor_type=SensorType.DS18B20,
    front_end_type=FrontEndType.DS18B20_1WIRE,
    sample_interval_seconds=1.0,
    raw_register_data={"scratchpad_crc_ok": False},
    detector_ruleset_version="ruleset-1",
    parameter_set_version="parameters-1",
)

flagged = record.with_quality_flags(
    QualityFlag(
        code="crc_failure",
        classification=QualityClass.IMPLAUSIBLE,
        message="Scratchpad CRC failed",
        detector="ds18b20_crc",
        detected_at=datetime.now(timezone.utc),
    )
)
```

The existing CSV pipeline does not yet provide all hardware provenance fields. Until acquisition is upgraded, missing hardware fields should remain empty or explicitly marked `unknown`; they must not be inferred as valid hardware state.

## Edge integrity payloads

The edge processor reads these optional `raw_register_data` fields when the corresponding hardware is present:

- MAX31855: `duplicate_read_word` or `duplicate_read_words`, `fault_summary`, `fault_open`, `fault_short_vcc`, `fault_short_gnd`.
- MAX31856: individual fault fields such as `open_thermocouple`, `over_voltage`, and `under_voltage`; `thermocouple_temperature`; `cold_junction_temperature`; `expected_configuration`; `configuration_readback`; and `configuration_reset`.
- DS18B20: `scratchpad_bytes`, `rom_id`, `expected_rom_id`, and `bus_rom_ids`.

MAX31855 reserved bits are checked using the documented reserved-bit mask, and the processor records that this interface has no CRC or parity. DS18B20 scratchpad CRC-8 is calculated locally. A configured `rom_manifests` map on `EdgeProcessor` verifies the complete bus topology at the configured periodic interval. None of these checks alter `raw_value`.

Timing fields are carried in `MeasurementRecord.timing`: monotonic conversion start/completion, declared conversion delay, conversion completion status, and explicit cross-sensor offset. `interval_statistics` carries count, mean, minimum, maximum, and standard deviation for raw oversamples; the individual raw samples remain the source data and are never replaced by the summary.
