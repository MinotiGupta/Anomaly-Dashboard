from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_layer import EdgeProcessor, EdgeStateStore
from measurement_contract import (
    FrontEndType,
    EventMetadata,
    MeasurementRecord,
    SensorType,
    TimingMetadata,
)
from physical_configuration import (
    ChannelPhysicalConfiguration,
    JunctionStyle,
    TauMeasurementMethod,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def configuration(**overrides):
    values = {
        "config_version": "physics-1",
        "device_id": "node-1",
        "board_id": "board-1",
        "node_id": "node-1",
        "channel_id": "channel-1",
        "sensor_type": SensorType.K_TYPE_THERMOCOUPLE,
        "front_end_type": FrontEndType.MAX31856,
        "junction_style": JunctionStyle.UNGROUNDED,
        "installation_location": "test rig",
        "thermal_time_constant_seconds": 10.0,
        "tau_uncertainty_seconds": 1.0,
        "tau_measurement_method": TauMeasurementMethod.BOTH,
        "tau_measured_at": "commissioning-2026-01-01",
        "minimum_physical_temperature_c": 0.0,
        "maximum_physical_temperature_c": 1000.0,
        "maximum_source_temperature_c": 1000.0,
        "minimum_sink_temperature_c": 0.0,
        "noise_floor_c": 0.1,
        "sample_interval_seconds": 1.0,
        "datasheet_resolution_c": 0.01,
        "thermocouple_tolerance_class": "Class_1",
    }
    values.update(overrides)
    return ChannelPhysicalConfiguration(**values)


def record(value, uptime, attributes=None, register_data=None):
    return MeasurementRecord(
        measurement_id=f"m-{uptime}-{value}",
        raw_value=value,
        wall_clock_timestamp=NOW,
        monotonic_uptime_seconds=uptime,
        boot_id="boot-1",
        device_id="node-1",
        channel_id="channel-1",
        sensor_type=SensorType.K_TYPE_THERMOCOUPLE,
        front_end_type=FrontEndType.MAX31856,
        sample_interval_seconds=1.0,
        detector_ruleset_version="ruleset-1",
        parameter_set_version="physics-1",
        event_metadata=EventMetadata(attributes=attributes or {}),
        raw_register_data=register_data or {},
        timing=TimingMetadata(),
    )


def codes(item):
    return {flag.code for flag in item.quality_flags}


def test_bounds_and_slope_are_applied_after_integrity(tmp_path):
    processor = EdgeProcessor(
        EdgeStateStore(tmp_path / "edge.db"),
        physical_configurations={"channel-1": configuration()},
    )
    processor.process(record(100.0, 1.0))
    flagged = processor.process(record(1000.0, 2.0))
    assert "thermal_slope_limit_exceeded" in codes(flagged)

    out_of_bounds = processor.process(record(-1.0, 3.0))
    assert "temperature_out_of_physical_bounds" in codes(out_of_bounds)


def test_cold_soak_and_high_pass_energy_checks(tmp_path):
    processor = EdgeProcessor(
        EdgeStateStore(tmp_path / "edge.db"),
        physical_configurations={"channel-1": configuration()},
    )
    cold = processor.process(
        record(25.0, 1.0, {"cold_soak_active": True, "cold_soak_reference_c": 20.0})
    )
    assert "cold_soak_self_check_failed" in codes(cold)

    for index in range(2, 5):
        noisy = processor.process(
            record(
                20.0 + index * 2.0,
                float(index),
                register_data={"high_pass_rms": 1.0},
            )
        )
    assert "high_pass_energy_exceeded" in codes(noisy)
    assert noisy.raw_value == 28.0


def test_two_sensor_lag_check_uses_explicit_offset(tmp_path):
    processor = EdgeProcessor(
        EdgeStateStore(tmp_path / "edge.db"),
        physical_configurations={"channel-1": configuration()},
    )
    flagged = processor.process(
        record(
            25.0,
            10.0,
            register_data={
                "peer_value": 20.0,
                "peer_monotonic_uptime_seconds": 1.0,
                "peer_expected_offset_seconds": 2.0,
            },
        )
    )
    assert "cross_sensor_timing_mismatch" in codes(flagged)


def test_tau_fit_and_gated_drift_are_stateful(tmp_path):
    processor = EdgeProcessor(
        EdgeStateStore(tmp_path / "edge.db"),
        physical_configurations={"channel-1": configuration()},
    )
    state = processor.state_store.load("node-1", "channel-1")
    state["physics_values"] = [0.0, 1.0, 0.0, 1.0]
    processor.state_store.save("node-1", "channel-1", state)
    drift = None
    for index in range(1, 30):
        drift = processor.process(
            record(
                20.0 + index * 0.05,
                float(index),
                attributes={"quiescent_window": True},
            )
        )
    assert "thermal_time_constant_drift" in codes(drift)
    assert "gated_drift_detected" in codes(drift)
