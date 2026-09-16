from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge_layer import EdgeProcessor, EdgeStateStore
from measurement_contract import FrontEndType, MeasurementRecord, SensorType


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_record(front_end, raw_value=20.0, raw_word=None, register_data=None):
    sensor_type = (
        SensorType.DS18B20
        if front_end is FrontEndType.DS18B20_1WIRE
        else SensorType.K_TYPE_THERMOCOUPLE
    )
    return MeasurementRecord(
        measurement_id=f"{front_end}-measurement",
        raw_value=raw_value,
        wall_clock_timestamp=NOW,
        monotonic_uptime_seconds=1.0,
        boot_id="boot-1",
        device_id="node-1",
        channel_id="channel-1",
        sensor_type=sensor_type,
        front_end_type=front_end,
        sample_interval_seconds=1.0,
        raw_hardware_word=raw_word,
        raw_register_data=register_data or {},
        detector_ruleset_version="ruleset-1",
        parameter_set_version="parameters-1",
    )


def codes(record):
    return {flag.code for flag in record.quality_flags}


def test_max31855_integrity_checks(tmp_path):
    processor = EdgeProcessor(EdgeStateStore(tmp_path / "edge.db"))
    record = make_record(
        FrontEndType.MAX31855,
        raw_word=(1 << 17) | (1 << 3),
        register_data={
            "duplicate_read_words": [0x12340000, 0x12340001],
            "fault_summary": True,
            "fault_open": False,
            "fault_short_vcc": False,
            "fault_short_gnd": False,
        },
    )

    flagged = processor.process(record)

    assert {
        "max31855_reserved_bits_set",
        "max31855_duplicate_reads_differ",
        "max31855_fault_bits_inconsistent",
    } <= codes(flagged)
    state = processor.state_store.load("node-1", "channel-1")
    assert state["integrity_capabilities"] == {"crc": False, "parity": False}


def test_max31856_faults_limits_and_configuration(tmp_path):
    processor = EdgeProcessor(EdgeStateStore(tmp_path / "edge.db"))
    flagged = processor.process(
        make_record(
            FrontEndType.MAX31856,
            register_data={
                "open_thermocouple": True,
                "over_voltage": True,
                "under_voltage": True,
                "thermocouple_temperature": 1900.0,
                "cold_junction_temperature": 100.0,
                "configuration_readback_ok": True,
                "expected_configuration": {"filter": 50},
                "configuration_readback": {"filter": 60},
            },
        )
    )

    assert {
        "max31856_open_thermocouple",
        "max31856_over_voltage",
        "max31856_under_voltage",
        "max31856_thermocouple_out_of_range",
        "max31856_cold_junction_out_of_range",
        "max31856_configuration_readback_mismatch",
    } <= codes(flagged)


def test_ds18b20_crc_rom_topology_and_failure_rate(tmp_path):
    processor = EdgeProcessor(
        EdgeStateStore(tmp_path / "edge.db"),
        rom_manifests={"node-1": {"rom-a"}},
        rom_manifest_check_interval=2,
    )
    valid_scratchpad = [0x50, 0x05, 0x4B, 0x46, 0x7F, 0xFF, 0x0C, 0x10, 0x1C]
    first = processor.process(
        make_record(
            FrontEndType.DS18B20_1WIRE,
            raw_value=85.0,
            raw_word=0,
            register_data={
                "scratchpad_bytes": valid_scratchpad,
                "rom_id": "rom-a",
                "expected_rom_id": "rom-a",
                "bus_rom_ids": ["rom-a"],
                "power_on_code": True,
                "conversion_completed": False,
            },
        )
    )
    assert "ds18b20_invalid_raw_word" in codes(first)
    assert "ds18b20_power_on_value" in codes(first)
    assert "ds18b20_crc_failure" not in codes(first)

    second = processor.process(
        make_record(
            FrontEndType.DS18B20_1WIRE,
            raw_word=0xFFFFFFFF,
            register_data={
                "scratchpad_bytes": valid_scratchpad[:-1] + [0x00],
                "rom_id": "rom-b",
                "expected_rom_id": "rom-a",
                "bus_rom_ids": ["rom-b"],
            },
        )
    )
    assert {
        "ds18b20_invalid_raw_word",
        "ds18b20_crc_failure",
        "ds18b20_rom_id_mismatch",
        "ds18b20_bus_topology_changed",
    } <= codes(second)
    state = processor.state_store.load("node-1", "channel-1")
    assert state["crc_failure_count"] == 1
    assert state["crc_sample_count"] == 2
    assert state["crc_failure_rate"] == 0.5
    assert state["bus_topology_change_count"] == 1
