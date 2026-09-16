"""Offline acquisition/edge processing for raw measurements."""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from measurement_contract import (
    FrontEndType,
    IntervalStatistics,
    MeasurementRecord,
    QualityClass,
    QualityFlag,
)
from physical_configuration import ChannelPhysicalConfiguration
from physics_detector import PhysicsDetector
from quality_policy import enforce_flag_authority
from thermal_bias import ThermalBiasAnalyzer


class EdgeStateStore:
    """Small durable store for detector state and unsent measurements."""

    def __init__(self, database_path: str | Path, outbox_path: str | Path | None = None):
        self.database_path = str(database_path)
        self.outbox_path = Path(outbox_path or f"{self.database_path}.outbox.jsonl")
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edge_state (
                device_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (device_id, channel_id)
            )
            """
        )
        self._connection.commit()

    def load(self, device_id: str, channel_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT state_json FROM edge_state WHERE device_id = ? AND channel_id = ?",
            (device_id, channel_id),
        ).fetchone()
        return json.loads(row[0]) if row else {}

    def save(self, device_id: str, channel_id: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO edge_state (device_id, channel_id, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(device_id, channel_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    device_id,
                    channel_id,
                    json.dumps(state, separators=(",", ":")),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._connection.commit()

    def enqueue(self, record: MeasurementRecord) -> None:
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.outbox_path.open("a", encoding="utf-8") as outbox:
            outbox.write(json.dumps(record.raw_payload(), separators=(",", ":")) + "\n")
            outbox.flush()

    def close(self) -> None:
        self._connection.close()


class EdgeProcessor:
    """Runs local integrity checks and emits raw records with additive flags."""

    MAX31855_RESERVED_MASK = (1 << 17) | (1 << 3)

    def __init__(
        self,
        state_store: EdgeStateStore,
        ruleset_version: str = "edge-1",
        max31856_limits: dict[str, float] | None = None,
        rom_manifests: dict[str, set[str] | list[str]] | None = None,
        rom_manifest_check_interval: int = 60,
        physical_configurations: dict[str, ChannelPhysicalConfiguration] | None = None,
    ):
        self.state_store = state_store
        self.ruleset_version = ruleset_version
        self.max31856_limits = max31856_limits or {
            "thermocouple_min": -270.0,
            "thermocouple_max": 1800.0,
            "cold_junction_min": -20.0,
            "cold_junction_max": 85.0,
        }
        self.rom_manifests = {
            device_id: set(rom_ids)
            for device_id, rom_ids in (rom_manifests or {}).items()
        }
        self.rom_manifest_check_interval = max(1, rom_manifest_check_interval)
        self.physical_configurations = physical_configurations or {}
        self.physics_detector = PhysicsDetector()
        self.thermal_bias_analyzer = ThermalBiasAnalyzer()

    def process(self, record: MeasurementRecord) -> MeasurementRecord:
        state = self.state_store.load(record.device_id, record.channel_id)
        flags = list(record.quality_flags)
        flags.extend(self._timing_flags(record, state))
        flags.extend(self._integrity_flags(record, state))
        configuration = self.physical_configurations.get(record.channel_id)
        if configuration is not None:
            flags.extend(
                self.physics_detector.evaluate(
                    record,
                    configuration,
                    state,
                    existing_flags=flags,
                )
            )
            diagnostics = self.thermal_bias_analyzer.analyze(record, configuration, state)
        else:
            diagnostics = []
        unique_flags = {flag.code: flag for flag in flags}
        authoritative_flags = enforce_flag_authority(unique_flags.values())
        flagged_record = record.model_copy(
            update={
                "quality_flags": authoritative_flags,
                "diagnostics": record.diagnostics + tuple(diagnostics),
            }
        )
        state.update(
            {
                "boot_id": record.boot_id,
                "last_measurement_id": record.measurement_id,
                "last_monotonic_uptime_seconds": record.monotonic_uptime_seconds,
                "last_wall_clock_timestamp": record.wall_clock_timestamp.isoformat(),
                "measurement_count": int(state.get("measurement_count", 0)) + 1,
                "flag_count": int(state.get("flag_count", 0)) + len(
                    authoritative_flags
                ),
                "ruleset_version": self.ruleset_version,
            }
        )
        state["boot_ids_seen"] = sorted(
            set(state.get("boot_ids_seen", [])) | {record.boot_id}
        )
        state["last_conversion_completed"] = record.timing.conversion_completed
        state["last_cross_sensor_offset_seconds"] = record.timing.cross_sensor_offset_seconds
        self._update_integrity_state(state, record, flagged_record)
        self.state_store.save(record.device_id, record.channel_id, state)
        self.state_store.enqueue(flagged_record)
        return flagged_record

    @staticmethod
    def summarize_interval(raw_values: list[float] | tuple[float, ...]) -> IntervalStatistics:
        """Summarize raw oversamples without changing or discarding them."""
        if not raw_values:
            raise ValueError("at least one raw oversample is required")
        values = [float(value) for value in raw_values]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("oversamples must be finite")
        return IntervalStatistics(
            sample_count=len(values),
            mean=statistics.fmean(values),
            minimum=min(values),
            maximum=max(values),
            standard_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
        )

    def _timing_flags(
        self, record: MeasurementRecord, state: dict[str, Any]
    ) -> list[QualityFlag]:
        flags = []
        previous_boot = state.get("boot_id")
        previous_uptime = state.get("last_monotonic_uptime_seconds")
        if previous_boot is not None and previous_boot != record.boot_id:
            flags.append(
                self._flag(
                    "boot_id_changed",
                    QualityClass.SUSPICIOUS,
                    "Device boot ID changed; detector state continuity crosses a restart",
                    "monotonic_timing",
                    record,
                )
            )
        if previous_boot == record.boot_id and previous_uptime is not None:
            delta = record.monotonic_uptime_seconds - float(previous_uptime)
            if delta <= 0:
                flags.append(
                    self._flag(
                        "monotonic_timestamp_regression",
                        QualityClass.IMPLAUSIBLE,
                        "Monotonic uptime did not advance within a boot session",
                        "monotonic_timing",
                        record,
                    )
                )
            elif abs(delta - record.sample_interval_seconds) > max(
                0.05, record.sample_interval_seconds * 0.25
            ):
                flags.append(
                    self._flag(
                        "sample_interval_mismatch",
                        QualityClass.SUSPICIOUS,
                        "Declared sample interval differs from monotonic elapsed time",
                        "monotonic_timing",
                        record,
                    )
                )
        timing = record.timing
        if (
            timing.conversion_started_monotonic_seconds is not None
            and timing.conversion_completed_monotonic_seconds is not None
            and timing.conversion_delay_seconds is not None
        ):
            actual_delay = (
                timing.conversion_completed_monotonic_seconds
                - timing.conversion_started_monotonic_seconds
            )
            if abs(actual_delay - timing.conversion_delay_seconds) > 0.01:
                flags.append(
                    self._flag(
                        "conversion_delay_mismatch",
                        QualityClass.SUSPICIOUS,
                        "Declared DS18B20 conversion delay differs from monotonic timing",
                        "conversion_timing",
                        record,
                    )
                )
        return flags

    def _integrity_flags(
        self, record: MeasurementRecord, state: dict[str, Any]
    ) -> list[QualityFlag]:
        if record.front_end_type is FrontEndType.MAX31855:
            return self._max31855_flags(record)
        if record.front_end_type is FrontEndType.MAX31856:
            return self._max31856_flags(record)
        if record.front_end_type is FrontEndType.DS18B20_1WIRE:
            return self._ds18b20_flags(record, state)
        return []

    @staticmethod
    def _max31855_flags(record: MeasurementRecord) -> list[QualityFlag]:
        word = record.raw_hardware_word
        if word is None:
            return []
        flags = []
        if word in (0, 0xFFFFFFFF):
            flags.append(
                EdgeProcessor._flag(
                    "max31855_invalid_word",
                    QualityClass.IMPLAUSIBLE,
                    "MAX31855 returned an all-zero or all-one data word",
                    "max31855_integrity",
                    record,
                )
            )
        if word & EdgeProcessor.MAX31855_RESERVED_MASK:
            flags.append(
                EdgeProcessor._flag(
                    "max31855_reserved_bits_set",
                    QualityClass.IMPLAUSIBLE,
                    "MAX31855 reserved bits are non-zero",
                    "max31855_integrity",
                    record,
                )
            )
        register_data = record.raw_register_data
        duplicate_word = register_data.get("duplicate_read_word")
        duplicate_words = register_data.get("duplicate_read_words")
        if duplicate_words is not None and isinstance(duplicate_words, (list, tuple)):
            words = list(duplicate_words)
            if len(words) >= 2 and len(set(words)) != 1:
                flags.append(
                    EdgeProcessor._flag(
                        "max31855_duplicate_reads_differ",
                        QualityClass.IMPLAUSIBLE,
                        "Closely spaced MAX31855 reads returned different words",
                        "max31855_duplicate_read",
                        record,
                    )
                )
        elif duplicate_word is not None and int(duplicate_word) != word:
            flags.append(
                EdgeProcessor._flag(
                    "max31855_duplicate_reads_differ",
                    QualityClass.IMPLAUSIBLE,
                    "Closely spaced MAX31855 reads returned different words",
                    "max31855_duplicate_read",
                    record,
                )
            )
        # MAX31855 supplies neither CRC nor parity; retain this fact in state.
        if register_data.get("fault_summary") is not None:
            specific_fault = any(
                bool(register_data.get(name))
                for name in ("fault_open", "fault_short_vcc", "fault_short_gnd")
            )
            if bool(register_data["fault_summary"]) != specific_fault:
                flags.append(
                    EdgeProcessor._flag(
                        "max31855_fault_bits_inconsistent",
                        QualityClass.IMPLAUSIBLE,
                        "MAX31855 summary and specific fault bits disagree",
                        "max31855_integrity",
                        record,
                    )
                )
        return flags

    def _max31856_flags(self, record: MeasurementRecord) -> list[QualityFlag]:
        flags = []
        register_data = record.raw_register_data
        fault_names = (
            "open_thermocouple",
            "over_voltage",
            "under_voltage",
            "thermocouple_high",
            "thermocouple_low",
            "cold_junction_high",
            "cold_junction_low",
            "cold_junction_range",
        )
        for fault_name in fault_names:
            if register_data.get(fault_name) is True:
                flags.append(
                    EdgeProcessor._flag(
                        f"max31856_{fault_name}",
                        QualityClass.IMPLAUSIBLE,
                        f"MAX31856 reported {fault_name.replace('_', ' ')}",
                        "max31856_fault_register",
                        record,
                    )
                )
        if (
            register_data.get("configuration_readback_ok") is False
            or register_data.get("configuration_reset") is True
        ):
            flags.append(
                EdgeProcessor._flag(
                    "max31856_configuration_mismatch",
                    QualityClass.IMPLAUSIBLE,
                    "MAX31856 configuration readback does not match expected state",
                    "max31856_configuration",
                    record,
                )
            )
        for value_name, minimum_name, maximum_name, label in (
            ("thermocouple_temperature", "thermocouple_min", "thermocouple_max", "thermocouple"),
            ("cold_junction_temperature", "cold_junction_min", "cold_junction_max", "cold junction"),
        ):
            value = register_data.get(value_name)
            if value is not None and (
                float(value) < self.max31856_limits[minimum_name]
                or float(value) > self.max31856_limits[maximum_name]
            ):
                flags.append(
                    EdgeProcessor._flag(
                        f"max31856_{label.replace(' ', '_')}_out_of_range",
                        QualityClass.IMPLAUSIBLE,
                        f"MAX31856 {label} temperature is outside configured limits",
                        "max31856_temperature_limits",
                        record,
                    )
                )
        expected_config = register_data.get("expected_configuration")
        actual_config = register_data.get("configuration_readback")
        if expected_config is not None and actual_config is not None and expected_config != actual_config:
            flags.append(
                EdgeProcessor._flag(
                    "max31856_configuration_readback_mismatch",
                    QualityClass.IMPLAUSIBLE,
                    "MAX31856 configuration readback differs from expected configuration",
                    "max31856_configuration",
                    record,
                )
            )
        return flags

    def _ds18b20_flags(
        self, record: MeasurementRecord, state: dict[str, Any]
    ) -> list[QualityFlag]:
        flags = []
        register_data = record.raw_register_data
        raw_word = record.raw_hardware_word
        if raw_word in (0, 0xFFFFFFFF):
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_invalid_raw_word",
                    QualityClass.IMPLAUSIBLE,
                    "DS18B20 returned an all-zero or all-one raw word",
                    "ds18b20_integrity",
                    record,
                )
            )
        scratchpad = register_data.get("scratchpad_bytes")
        if isinstance(scratchpad, (list, tuple)) and len(scratchpad) >= 9:
            expected_crc = int(scratchpad[8])
            calculated_crc = EdgeProcessor._ds18b20_crc8(scratchpad[:8])
            if calculated_crc != expected_crc:
                register_data_crc_ok = False
            else:
                register_data_crc_ok = True
        else:
            register_data_crc_ok = register_data.get("crc_ok")
        if register_data_crc_ok is False:
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_crc_failure",
                    QualityClass.IMPLAUSIBLE,
                    "DS18B20 scratchpad CRC failed",
                    "ds18b20_crc",
                    record,
                )
            )
        if register_data.get("bus_present") is False:
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_bus_missing",
                    QualityClass.IMPLAUSIBLE,
                    "DS18B20 presence pulse was not detected",
                    "ds18b20_bus",
                    record,
                )
            )
        if register_data.get("rom_id_match") is False:
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_rom_id_mismatch",
                    QualityClass.IMPLAUSIBLE,
                    "DS18B20 ROM ID differs from the expected device manifest",
                    "ds18b20_rom_manifest",
                    record,
                )
            )
        if (
            register_data.get("expected_rom_id") is not None
            and register_data.get("rom_id") != register_data.get("expected_rom_id")
        ):
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_rom_id_mismatch",
                    QualityClass.IMPLAUSIBLE,
                    "DS18B20 ROM ID differs from the expected device manifest",
                    "ds18b20_rom_manifest",
                    record,
                )
            )
        if register_data.get("bus_topology_changed") is True:
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_bus_topology_changed",
                    QualityClass.IMPLAUSIBLE,
                    "DS18B20 ROM-ID bus topology changed from the expected manifest",
                    "ds18b20_rom_manifest",
                    record,
                )
            )
        manifest = self.rom_manifests.get(record.device_id)
        sample_number = int(state.get("measurement_count", 0)) + 1
        current_bus = register_data.get("bus_rom_ids")
        if (
            manifest is not None
            and isinstance(current_bus, (list, tuple, set))
            and sample_number % self.rom_manifest_check_interval == 0
            and set(current_bus) != manifest
        ):
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_bus_topology_changed",
                    QualityClass.IMPLAUSIBLE,
                    "Periodic DS18B20 ROM-ID manifest verification failed",
                    "ds18b20_rom_manifest",
                    record,
                )
            )
        if (
            record.raw_value == 85.0
            and register_data.get("power_on_code") is True
            and register_data.get("conversion_completed") is False
        ):
            flags.append(
                EdgeProcessor._flag(
                    "ds18b20_power_on_value",
                    QualityClass.SUSPICIOUS,
                    "Exact DS18B20 power-on value observed before conversion completed",
                    "ds18b20_conversion_state",
                    record,
                )
            )
        return flags

    def _update_integrity_state(
        self,
        state: dict[str, Any],
        record: MeasurementRecord,
        flagged_record: MeasurementRecord,
    ) -> None:
        register_data = record.raw_register_data
        state.setdefault("integrity_capabilities", {})
        if record.front_end_type is FrontEndType.MAX31855:
            state["integrity_capabilities"].update({"crc": False, "parity": False})
        if record.front_end_type is FrontEndType.DS18B20_1WIRE:
            crc_failed = any(flag.code == "ds18b20_crc_failure" for flag in flagged_record.quality_flags)
            state["crc_failure_count"] = int(state.get("crc_failure_count", 0)) + int(crc_failed)
            state["crc_sample_count"] = int(state.get("crc_sample_count", 0)) + 1
            state["crc_failure_rate"] = state["crc_failure_count"] / state["crc_sample_count"]
            current_rom = register_data.get("rom_id")
            expected_rom = register_data.get("expected_rom_id")
            if expected_rom is not None and current_rom != expected_rom:
                state["rom_manifest_mismatch_count"] = int(state.get("rom_manifest_mismatch_count", 0)) + 1
            previous_rom = state.get("last_rom_id")
            if previous_rom is not None and current_rom is not None and previous_rom != current_rom:
                state["bus_topology_change_count"] = int(state.get("bus_topology_change_count", 0)) + 1
            state["last_rom_id"] = current_rom
            state["expected_rom_id"] = expected_rom
            state["last_bus_topology"] = register_data.get("bus_rom_ids", state.get("last_bus_topology"))

    @staticmethod
    def _ds18b20_crc8(bytes_to_check: list[int] | tuple[int, ...]) -> int:
        crc = 0
        for byte in bytes_to_check:
            value = int(byte) & 0xFF
            for _ in range(8):
                mix = (crc ^ value) & 0x01
                crc >>= 1
                if mix:
                    crc ^= 0x8C
                value >>= 1
        return crc

    @staticmethod
    def _flag(
        code: str,
        classification: QualityClass,
        message: str,
        detector: str,
        record: MeasurementRecord,
    ) -> QualityFlag:
        return QualityFlag(
            code=code,
            classification=classification,
            message=message,
            detector=detector,
            detected_at=record.wall_clock_timestamp,
        )
