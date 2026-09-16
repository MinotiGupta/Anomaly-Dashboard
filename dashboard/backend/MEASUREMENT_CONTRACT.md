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

## Per-channel physical configuration

`physical_configuration.py` defines the channel configuration published to an edge node. It includes sensor and front-end identity, junction style, probe geometry and installation, measured thermal time constant and uncertainty, physical bounds, source/sink limits, noise floor, acquisition interval, datasheet resolution, thermocouple tolerance class, board/node identity, and related fast/slow sensor pairs.

The model accepts only measured-$\\tau$ provenance (`commissioning_step`, `online_transient_fit`, or both). A catalogue default is intentionally invalid. Publish it through `PUT /api/devices/<device_id>/channels/<channel_id>/physical-config`; retrieve the active version with the corresponding `GET` endpoint.

## Physics detector order

`physics_detector.py` applies the PDF sequence after the existing hardware signal-integrity checks:

1. Thermodynamic bounds using channel and source/sink limits.
2. Per-channel slope limit using measured `thermal_time_constant_seconds` and its uncertainty.
3. Cold-soak agreement when `event_metadata.attributes` contains `cold_soak_active` and `cold_soak_reference_c`.
4. High-pass RMS or supplied `raw_register_data["high_pass_rms"]`, compared with the configured noise floor and resolution.
5. Two-sensor timing/residual checks using explicit peer timing metadata or a peer record.
6. Online first-order time-constant fitting and drift annotation.
7. CUSUM-like drift accumulation only inside explicitly marked `quiescent_window` intervals.

The edge layer runs signal-integrity checks first, then adds these physics flags without changing `raw_value`. Hard violations are `implausible`; degradation signatures are `suspicious`; high-pass or residual surprises that remain physically possible are `unusual` annotations.
