from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_layer import EdgeProcessor, EdgeStateStore
from measurement_contract import (
    FrontEndType,
    IntervalStatistics,
    MeasurementRecord,
    SensorType,
    TimingMetadata,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def record(measurement_id, uptime, boot_id="boot-1", interval=1.0, timing=None):
    return MeasurementRecord(
        measurement_id=measurement_id,
        raw_value=20.0,
        wall_clock_timestamp=NOW,
        monotonic_uptime_seconds=uptime,
        boot_id=boot_id,
        device_id="node-1",
        channel_id="channel-1",
        sensor_type=SensorType.DS18B20,
        front_end_type=FrontEndType.DS18B20_1WIRE,
        sample_interval_seconds=interval,
        detector_ruleset_version="ruleset-1",
        parameter_set_version="parameters-1",
        timing=timing or TimingMetadata(),
    )


def codes(item):
    return {flag.code for flag in item.quality_flags}


def test_monotonic_timing_and_restart_state_persist(tmp_path):
    store = EdgeStateStore(tmp_path / "edge.db")
    processor = EdgeProcessor(store)
    processor.process(record("m1", 1.0))
    second = processor.process(record("m2", 2.0))
    restarted = processor.process(record("m3", 0.1, boot_id="boot-2"))

    assert not {"monotonic_timestamp_regression", "sample_interval_mismatch"} & codes(second)
    assert "boot_id_changed" in codes(restarted)
    state = store.load("node-1", "channel-1")
    assert state["measurement_count"] == 3
    assert state["boot_ids_seen"] == ["boot-1", "boot-2"]
    store.close()


def test_monotonic_regression_and_interval_mismatch_are_flagged(tmp_path):
    processor = EdgeProcessor(EdgeStateStore(tmp_path / "edge.db"))
    processor.process(record("m1", 10.0))
    flagged = processor.process(record("m2", 9.0))
    assert "monotonic_timestamp_regression" in codes(flagged)

    processor.process(record("m3", 10.0))
    mismatch = processor.process(record("m4", 12.0, interval=1.0))
    assert "sample_interval_mismatch" in codes(mismatch)


def test_conversion_delay_and_cross_sensor_offset_are_explicit():
    timing = TimingMetadata(
        conversion_started_monotonic_seconds=10.0,
        conversion_completed_monotonic_seconds=10.75,
        conversion_delay_seconds=0.5,
        cross_sensor_offset_seconds=0.75,
    )
    record_value = record("m1", 11.0, timing=timing)
    assert record_value.timing.cross_sensor_offset_seconds == 0.75
    assert record_value.timing.conversion_delay_seconds == 0.5


def test_conversion_timing_mismatch_is_flagged(tmp_path):
    processor = EdgeProcessor(EdgeStateStore(tmp_path / "edge.db"))
    timing = TimingMetadata(
        conversion_started_monotonic_seconds=10.0,
        conversion_completed_monotonic_seconds=10.75,
        conversion_delay_seconds=0.5,
    )
    flagged = processor.process(record("m1", 11.0, timing=timing))
    assert "conversion_delay_mismatch" in codes(flagged)


def test_oversampled_interval_statistics_are_preserved_and_validated():
    summary = EdgeProcessor.summarize_interval([1.0, 2.0, 3.0, 4.0])
    assert summary == IntervalStatistics(
        sample_count=4,
        mean=2.5,
        minimum=1.0,
        maximum=4.0,
        standard_deviation=1.2909944487358056,
    )
    with pytest.raises(ValueError):
        EdgeProcessor.summarize_interval([])
