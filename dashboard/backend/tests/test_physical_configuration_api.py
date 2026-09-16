import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as backend_app
from dashboard_store import DashboardStore


def test_physical_configuration_routes_validate_and_publish(tmp_path):
    backend_app.dashboard_store = DashboardStore(tmp_path / "api.db")
    client = backend_app.app.test_client()
    payload = {
        "config_version": "physical-api-1",
        "board_id": "board-api",
        "node_id": "node-api",
        "sensor_type": "k_type_thermocouple",
        "front_end_type": "max31856",
        "junction_style": "ungrounded",
        "probe_diameter_mm": 1.5,
        "installation_location": "test rig",
        "insertion_depth_mm": 20.0,
        "thermal_time_constant_seconds": 10.0,
        "tau_uncertainty_seconds": 1.0,
        "tau_measurement_method": "commissioning_step",
        "tau_measured_at": "2026-09-17T00:00:00Z",
        "minimum_physical_temperature_c": -20.0,
        "maximum_physical_temperature_c": 1370.0,
        "maximum_source_temperature_c": 2400.0,
        "minimum_sink_temperature_c": 20.0,
        "noise_floor_c": 0.1,
        "sample_interval_seconds": 1.0,
        "datasheet_resolution_c": 0.0078,
        "thermocouple_tolerance_class": "IEC60584_Class_1",
        "related_sensor_pairs": [],
    }
    response = client.put(
        "/api/devices/node-api/channels/channel-api/physical-config",
        json=payload,
    )
    assert response.status_code == 201
    assert response.json["channel_id"] == "channel-api"

    response = client.get("/api/devices/node-api/channels/channel-api/physical-config")
    assert response.status_code == 200
    assert response.json["thermal_time_constant_seconds"] == 10.0

    response = client.get("/api/devices/node-api/physical-config")
    assert response.status_code == 200
    assert len(response.json["channels"]) == 1
