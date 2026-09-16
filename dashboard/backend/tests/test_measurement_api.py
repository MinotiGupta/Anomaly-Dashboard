from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as backend_app
from dashboard_store import DashboardStore
from measurement_contract import FrontEndType, MeasurementRecord, SensorType


def test_measurement_and_configuration_routes(tmp_path):
    backend_app.dashboard_store = DashboardStore(tmp_path / "api.db")
    client = backend_app.app.test_client()
    record = MeasurementRecord(
        measurement_id="api-measurement-1",
        raw_value=22.5,
        wall_clock_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        monotonic_uptime_seconds=1.0,
        boot_id="boot-api",
        device_id="node-api",
        channel_id="channel-api",
        sensor_type=SensorType.K_TYPE_THERMOCOUPLE,
        front_end_type=FrontEndType.MAX31856,
        sample_interval_seconds=1.0,
        detector_ruleset_version="ruleset-api",
        parameter_set_version="parameters-api",
    )

    response = client.post(
        "/api/measurements",
        json={"run_id": "run-api", "records": [record.raw_payload()]},
    )
    assert response.status_code == 202
    assert response.json["stored"] == 1

    response = client.get("/api/measurements/node-api")
    assert response.status_code == 200
    assert response.json["records"][0]["raw_value"] == 22.5

    response = client.put(
        "/api/edge-config/node-api",
        json={
            "config_version": "config-api",
            "ruleset_version": "ruleset-api-2",
            "parameter_set_version": "parameters-api-2",
            "configuration": {"channels": {"channel-api": {"tau_seconds": 3.0}}},
        },
    )
    assert response.status_code == 201

    response = client.get("/api/edge-config/node-api")
    assert response.status_code == 200
    assert response.json["config_version"] == "config-api"
