from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_store import DashboardStore
from edge_layer import EdgeProcessor, EdgeStateStore
from measurement_contract import (
    FrontEndType,
    MeasurementRecord,
    SensorType,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_record(measurement_id="measurement-1", raw_value=85.0, uptime=1.0):
    return MeasurementRecord(
        measurement_id=measurement_id,
        raw_value=raw_value,
        wall_clock_timestamp=NOW,
        monotonic_uptime_seconds=uptime,
        boot_id="boot-1",
        device_id="node-1",
        channel_id="probe-1",
        sensor_type=SensorType.DS18B20,
        front_end_type=FrontEndType.DS18B20_1WIRE,
        sample_interval_seconds=1.0,
        raw_register_data={
            "crc_ok": False,
            "power_on_code": True,
            "conversion_completed": False,
        },
        detector_ruleset_version="ruleset-1",
        parameter_set_version="parameters-1",
    )


def test_edge_flags_and_persists_state(tmp_path):
    state_store = EdgeStateStore(tmp_path / "edge.db")
    processor = EdgeProcessor(state_store)

    flagged = processor.process(make_record())
    codes = {flag.code for flag in flagged.quality_flags}

    assert flagged.raw_value == 85.0
    assert {"ds18b20_crc_failure", "ds18b20_power_on_value"} <= codes
    assert state_store.load("node-1", "probe-1")["measurement_count"] == 1
    assert state_store.outbox_path.read_text(encoding="utf-8").count("measurement-1") == 1

    state_store.close()
    reopened = EdgeStateStore(tmp_path / "edge.db")
    assert reopened.load("node-1", "probe-1")["last_measurement_id"] == "measurement-1"
    reopened.close()


def test_dashboard_stores_raw_records_and_versioned_config(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.db")
    raw_records = [
        make_record("measurement-1", raw_value=20.0, uptime=1.0),
        make_record("measurement-2", raw_value=21.0, uptime=2.0),
        make_record("measurement-3", raw_value=20.5, uptime=3.0),
    ]
    edge_store = EdgeStateStore(tmp_path / "edge.db")
    records = [EdgeProcessor(edge_store).process(record) for record in raw_records]

    assert store.ingest_many(records, run_id="run-1") == 3
    assert store.ingest(records[0], run_id="run-1") is False
    stored = store.list_measurements("node-1")
    assert [record.raw_value for record in stored] == [20.0, 21.0, 20.5]
    assert store.diagnostics("node-1")["flag_count"] == 3
    assert store.fit_channel_parameters("node-1")[0]["sample_count"] == 3

    store.save_configuration(
        "node-1",
        "config-2",
        "ruleset-2",
        "parameters-2",
        {"channels": {"probe-1": {"tau_seconds": 5.0}}},
    )
    config = store.get_configuration("node-1")
    assert config["config_version"] == "config-2"
    assert config["configuration"]["channels"]["probe-1"]["tau_seconds"] == 5.0
