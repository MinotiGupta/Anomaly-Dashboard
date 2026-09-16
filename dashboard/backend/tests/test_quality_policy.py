from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_store import DashboardStore
from measurement_contract import (
    FrontEndType,
    MeasurementRecord,
    QualityClass,
    QualityFlag,
    SensorType,
)
from quality_policy import enforce_flag_authority, quality_summary, science_view


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def flag(code, classification, detector):
    return QualityFlag(
        code=code,
        classification=classification,
        message=code,
        detector=detector,
        detected_at=NOW,
    )


def record(measurement_id, flags=()):
    return MeasurementRecord(
        measurement_id=measurement_id,
        raw_value=20.0,
        wall_clock_timestamp=NOW,
        monotonic_uptime_seconds=1.0,
        boot_id="boot-1",
        device_id="node-1",
        channel_id="channel-1",
        sensor_type=SensorType.K_TYPE_THERMOCOUPLE,
        front_end_type=FrontEndType.MAX31856,
        sample_interval_seconds=1.0,
        detector_ruleset_version="ruleset-1",
        parameter_set_version="parameters-1",
        quality_flags=tuple(flags),
    )


def test_statistical_detector_can_only_annotate_as_unusual():
    normalized = enforce_flag_authority(
        [flag("reconstruction_error", QualityClass.IMPLAUSIBLE, "autoencoder")]
    )
    assert normalized[0].classification is QualityClass.UNUSUAL


def test_quality_summary_and_science_view_preserve_raw_records():
    implausible = record(
        "bad",
        [flag("out_of_bounds", QualityClass.IMPLAUSIBLE, "thermodynamic_bounds")],
    )
    suspicious = record(
        "suspect",
        [flag("tau_drift", QualityClass.SUSPICIOUS, "online_tau_fit")],
    )
    unusual = record(
        "unusual",
        [flag("high_pass", QualityClass.UNUSUAL, "high_pass_energy")],
    )

    assert quality_summary(implausible) == {
        "implausible": 1,
        "suspicious": 0,
        "unusual": 0,
        "science_excluded": True,
        "raw_retained": True,
    }
    selected = science_view([implausible, suspicious, unusual])
    assert [item.measurement_id for item in selected] == ["suspect", "unusual"]
    assert implausible.raw_value == 20.0


def test_dashboard_keeps_raw_and_exposes_quality_summary(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.db")
    bad = record(
        "bad",
        [flag("out_of_bounds", QualityClass.IMPLAUSIBLE, "thermodynamic_bounds")],
    )
    good = record(
        "good",
        [flag("drift", QualityClass.SUSPICIOUS, "online_tau_fit")],
    )
    assert store.ingest_many([bad, good]) == 2
    assert len(store.list_measurements("node-1")) == 2
    assert len(store.list_science_measurements("node-1")) == 1
    assert store.diagnostics("node-1")["raw_records_retained"] is True
    assert store.diagnostics("node-1")["quality_summary"] == {
        "implausible": 1,
        "suspicious": 1,
        "unusual": 0,
    }
