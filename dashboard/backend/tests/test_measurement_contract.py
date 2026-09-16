from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measurement_contract import (
    FrontEndType,
    MeasurementRecord,
    QualityClass,
    QualityFlag,
    SensorType,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_record(**overrides):
    values = {
        "measurement_id": "measurement-1",
        "raw_value": 85.0,
        "wall_clock_timestamp": NOW,
        "monotonic_uptime_seconds": 10.0,
        "boot_id": "boot-1",
        "device_id": "node-1",
        "channel_id": "channel-1",
        "sensor_type": SensorType.DS18B20,
        "front_end_type": FrontEndType.DS18B20_1WIRE,
        "sample_interval_seconds": 1.0,
        "detector_ruleset_version": "ruleset-1",
        "parameter_set_version": "parameters-1",
    }
    values.update(overrides)
    return MeasurementRecord(**values)


def make_flag(code="crc_failure"):
    return QualityFlag(
        code=code,
        classification=QualityClass.IMPLAUSIBLE,
        message="Scratchpad CRC failed",
        detector="ds18b20_crc",
        detected_at=NOW,
    )


def test_record_contains_raw_and_provenance_fields():
    record = make_record(
        raw_hardware_word=0x12345678,
        raw_register_data={"fault_status": 4},
        event_metadata={"igniter_state": "off"},
    )

    payload = record.raw_payload()

    assert payload["schema_version"] == "1.0"
    assert payload["raw_value"] == 85.0
    assert payload["raw_hardware_word"] == 0x12345678
    assert payload["raw_register_data"]["fault_status"] == 4
    assert payload["event_metadata"]["igniter_state"] == "off"


def test_detector_flags_are_additive_and_raw_record_is_unchanged():
    record = make_record()
    flagged = record.with_quality_flags(make_flag())

    assert record.raw_value == 85.0
    assert record.quality_flags == ()
    assert flagged.raw_value == record.raw_value
    assert flagged.measurement_id == record.measurement_id
    assert [flag.code for flag in flagged.quality_flags] == ["crc_failure"]


def test_duplicate_quality_codes_are_rejected():
    with pytest.raises(ValidationError, match="duplicate codes"):
        make_record(quality_flags=(make_flag(), make_flag()))


@pytest.mark.parametrize(
    "field, value",
    [
        ("wall_clock_timestamp", datetime(2026, 1, 1)),
        ("monotonic_uptime_seconds", -1.0),
        ("sample_interval_seconds", 0.0),
    ],
)
def test_timing_constraints_are_enforced(field, value):
    with pytest.raises(ValidationError):
        make_record(**{field: value})
