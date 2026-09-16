from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_store import DashboardStore
from measurement_contract import FrontEndType, SensorType
from physical_configuration import (
    ChannelPhysicalConfiguration,
    JunctionStyle,
    SensorPair,
    TauMeasurementMethod,
)


def make_configuration(**overrides):
    values = {
        "config_version": "physical-1",
        "device_id": "node-1",
        "board_id": "board-1",
        "node_id": "node-1",
        "channel_id": "tc-1",
        "sensor_type": SensorType.K_TYPE_THERMOCOUPLE,
        "front_end_type": FrontEndType.MAX31856,
        "junction_style": JunctionStyle.UNGROUNDED,
        "probe_diameter_mm": 1.5,
        "installation_location": "combustor inlet",
        "insertion_depth_mm": 30.0,
        "thermal_time_constant_seconds": 12.0,
        "tau_uncertainty_seconds": 1.5,
        "tau_measurement_method": TauMeasurementMethod.COMMISSIONING_STEP,
        "tau_measured_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "minimum_physical_temperature_c": -20.0,
        "maximum_physical_temperature_c": 1370.0,
        "maximum_source_temperature_c": 2400.0,
        "minimum_sink_temperature_c": 20.0,
        "noise_floor_c": 0.08,
        "sample_interval_seconds": 1.0,
        "datasheet_resolution_c": 0.0078,
        "thermocouple_tolerance_class": "IEC60584_Class_1",
        "related_sensor_pairs": (
            SensorPair(channel_id="tc-2", relationship="slow", expected_offset_seconds=2.0),
        ),
    }
    values.update(overrides)
    return ChannelPhysicalConfiguration(**values)


def test_configuration_requires_measured_tau_and_contains_all_physical_fields():
    configuration = make_configuration()
    payload = configuration.as_edge_payload()
    assert payload["thermal_time_constant_seconds"] == 12.0
    assert payload["tau_measurement_method"] == "commissioning_step"
    assert payload["related_sensor_pairs"][0]["relationship"] == "slow"


def test_catalogue_tau_provenance_is_not_accepted():
    with pytest.raises(ValidationError):
        make_configuration(tau_measurement_method="catalogue_default")


def test_configuration_is_persisted_and_retrieved(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.db")
    configuration = make_configuration()
    store.save_channel_configuration(configuration)
    loaded = store.get_channel_configuration("node-1", "tc-1")
    assert loaded == configuration
    assert store.list_channel_configurations("node-1") == [configuration]
