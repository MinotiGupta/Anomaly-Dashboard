from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_layer import EdgeProcessor, EdgeStateStore
from measurement_contract import FrontEndType, MeasurementRecord, SensorType
from physical_configuration import ChannelPhysicalConfiguration, JunctionStyle, TauMeasurementMethod


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def configuration():
    return ChannelPhysicalConfiguration(
        config_version="bias-1",
        device_id="node-1",
        board_id="board-1",
        node_id="node-1",
        channel_id="channel-1",
        sensor_type=SensorType.K_TYPE_THERMOCOUPLE,
        front_end_type=FrontEndType.MAX31855,
        junction_style=JunctionStyle.GROUNDED,
        probe_diameter_mm=1.5,
        installation_location="combustor",
        insertion_depth_mm=10.0,
        thermal_time_constant_seconds=10.0,
        tau_uncertainty_seconds=1.0,
        tau_measurement_method=TauMeasurementMethod.BOTH,
        tau_measured_at="commissioning",
        minimum_physical_temperature_c=0.0,
        maximum_physical_temperature_c=1370.0,
        maximum_source_temperature_c=2400.0,
        minimum_sink_temperature_c=20.0,
        noise_floor_c=0.1,
        sample_interval_seconds=1.0,
        datasheet_resolution_c=0.25,
        thermocouple_tolerance_class="Class_1",
    )


def record(register_data):
    return MeasurementRecord(
        measurement_id="bias-1",
        raw_value=800.0,
        wall_clock_timestamp=NOW,
        monotonic_uptime_seconds=1.0,
        boot_id="boot-1",
        device_id="node-1",
        channel_id="channel-1",
        sensor_type=SensorType.K_TYPE_THERMOCOUPLE,
        front_end_type=FrontEndType.MAX31855,
        sample_interval_seconds=1.0,
        raw_register_data=register_data,
        detector_ruleset_version="ruleset-1",
        parameter_set_version="bias-1",
    )


def test_expected_biases_are_annotations_not_quality_flags(tmp_path):
    processor = EdgeProcessor(
        EdgeStateStore(tmp_path / "edge.db"),
        physical_configurations={"channel-1": configuration()},
    )
    first = processor.process(
        record(
            {
                "wall_temperature_c": 300.0,
                "heat_transfer_coefficient_w_m2k": 1000.0,
                "emissivity": 0.6,
                "mount_temperature_c": 400.0,
                "gas_velocity_m_s": 100.0,
                "max31855_linearisation_difference_c": 2.0,
                "wire_repositioned": True,
                "board_temperature_c": 30.0,
            }
        )
    )
    second = processor.process(
        record({"board_temperature_c": 32.0})
    )

    codes = {annotation.code for annotation in first.diagnostics}
    assert {
        "radiation_error_expected",
        "conduction_error_expected",
        "recovery_effect_expected",
        "thermocouple_inhomogeneity_diagnostic",
        "junction_style_diagnostic",
    } <= codes
    assert "cold_junction_drift_diagnostic" in second.diagnostics[0].code or any(
        item.code == "cold_junction_drift_diagnostic" for item in second.diagnostics
    )
    assert not first.quality_flags
    assert first.raw_value == 800.0


def test_bias_annotations_serialize_separately():
    payload = record({"max31855_linearisation_difference_c": 1.5}).raw_payload()
    assert payload["quality_flags"] == []
    assert payload["diagnostics"] == []
