"""Offline acquisition/edge processing for raw measurements."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from measurement_contract import (
    FrontEndType,
    MeasurementRecord,
    QualityClass,
    QualityFlag,
)


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

    def __init__(self, state_store: EdgeStateStore, ruleset_version: str = "edge-1"):
        self.state_store = state_store
        self.ruleset_version = ruleset_version

    def process(self, record: MeasurementRecord) -> MeasurementRecord:
        state = self.state_store.load(record.device_id, record.channel_id)
        flags = list(record.quality_flags)
        flags.extend(self._integrity_flags(record))
        unique_flags = {flag.code: flag for flag in flags}
        flagged_record = record.model_copy(
            update={"quality_flags": tuple(unique_flags.values())}
        )
        state.update(
            {
                "boot_id": record.boot_id,
                "last_measurement_id": record.measurement_id,
                "last_monotonic_uptime_seconds": record.monotonic_uptime_seconds,
                "last_wall_clock_timestamp": record.wall_clock_timestamp.isoformat(),
                "measurement_count": int(state.get("measurement_count", 0)) + 1,
                "flag_count": int(state.get("flag_count", 0)) + len(
                    tuple(unique_flags.values())
                ),
                "ruleset_version": self.ruleset_version,
            }
        )
        self.state_store.save(record.device_id, record.channel_id, state)
        self.state_store.enqueue(flagged_record)
        return flagged_record

    def _integrity_flags(self, record: MeasurementRecord) -> list[QualityFlag]:
        if record.front_end_type is FrontEndType.MAX31855:
            return self._max31855_flags(record)
        if record.front_end_type is FrontEndType.MAX31856:
            return self._max31856_flags(record)
        if record.front_end_type is FrontEndType.DS18B20_1WIRE:
            return self._ds18b20_flags(record)
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
        register_data = record.raw_register_data
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

    @staticmethod
    def _max31856_flags(record: MeasurementRecord) -> list[QualityFlag]:
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
        return flags

    @staticmethod
    def _ds18b20_flags(record: MeasurementRecord) -> list[QualityFlag]:
        flags = []
        register_data = record.raw_register_data
        if register_data.get("crc_ok") is False:
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
