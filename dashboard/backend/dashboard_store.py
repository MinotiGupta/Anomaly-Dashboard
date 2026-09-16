"""Durable dashboard-side storage, diagnostics, and edge configuration registry."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from measurement_contract import MeasurementRecord


class DashboardStore:
    """SQLite-backed cloud/dashboard boundary for validated edge records."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        connection = sqlite3.connect(self.database_path)
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id TEXT PRIMARY KEY,
                run_id TEXT,
                device_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                wall_clock_timestamp TEXT NOT NULL,
                raw_value REAL NOT NULL,
                payload_json TEXT NOT NULL,
                received_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_measurements_channel
                ON measurements(device_id, channel_id, wall_clock_timestamp);
            CREATE TABLE IF NOT EXISTS edge_configurations (
                device_id TEXT NOT NULL,
                config_version TEXT NOT NULL,
                ruleset_version TEXT NOT NULL,
                parameter_set_version TEXT NOT NULL,
                configuration_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (device_id, config_version)
            );
            """
        )
        connection.commit()
        connection.close()

    def ingest(self, record: MeasurementRecord, run_id: str | None = None) -> bool:
        """Store a validated record without changing its raw payload."""
        payload = record.raw_payload()
        connection = sqlite3.connect(self.database_path)
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO measurements (
                measurement_id, run_id, device_id, channel_id,
                wall_clock_timestamp, raw_value, payload_json, received_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.measurement_id,
                run_id,
                record.device_id,
                record.channel_id,
                record.wall_clock_timestamp.isoformat(),
                float(record.raw_value),
                json.dumps(payload, separators=(",", ":")),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()
        inserted = cursor.rowcount == 1
        connection.close()
        return inserted

    def ingest_many(
        self, records: Iterable[MeasurementRecord], run_id: str | None = None
    ) -> int:
        return sum(self.ingest(record, run_id=run_id) for record in records)

    def list_measurements(
        self, device_id: str, channel_id: str | None = None, limit: int = 1000
    ) -> list[MeasurementRecord]:
        query = "SELECT payload_json FROM measurements WHERE device_id = ?"
        parameters: list[Any] = [device_id]
        if channel_id is not None:
            query += " AND channel_id = ?"
            parameters.append(channel_id)
        query += " ORDER BY wall_clock_timestamp ASC LIMIT ?"
        parameters.append(max(1, min(limit, 10000)))
        connection = sqlite3.connect(self.database_path)
        rows = connection.execute(query, parameters).fetchall()
        connection.close()
        return [MeasurementRecord.model_validate(json.loads(row[0])) for row in rows]

    def diagnostics(self, device_id: str) -> dict[str, Any]:
        records = self.list_measurements(device_id)
        flags: list[dict[str, Any]] = []
        for record in records:
            for flag in record.quality_flags:
                flags.append(
                    {
                        "measurement_id": record.measurement_id,
                        "channel_id": record.channel_id,
                        "timestamp": record.wall_clock_timestamp.isoformat(),
                        **flag.model_dump(mode="json"),
                    }
                )
        return {
            "device_id": device_id,
            "measurement_count": len(records),
            "flag_count": len(flags),
            "flags": flags,
        }

    def save_configuration(
        self,
        device_id: str,
        config_version: str,
        ruleset_version: str,
        parameter_set_version: str,
        configuration: dict[str, Any],
    ) -> None:
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            """
            INSERT OR REPLACE INTO edge_configurations (
                device_id, config_version, ruleset_version,
                parameter_set_version, configuration_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                device_id,
                config_version,
                ruleset_version,
                parameter_set_version,
                json.dumps(configuration, separators=(",", ":")),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()
        connection.close()

    def get_configuration(self, device_id: str, config_version: str | None = None):
        connection = sqlite3.connect(self.database_path)
        if config_version is None:
            row = connection.execute(
                """
                SELECT config_version, ruleset_version, parameter_set_version,
                       configuration_json, created_at
                FROM edge_configurations
                WHERE device_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (device_id,),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT config_version, ruleset_version, parameter_set_version,
                       configuration_json, created_at
                FROM edge_configurations
                WHERE device_id = ? AND config_version = ?
                """,
                (device_id, config_version),
            ).fetchone()
        connection.close()
        if row is None:
            return None
        return {
            "device_id": device_id,
            "config_version": row[0],
            "ruleset_version": row[1],
            "parameter_set_version": row[2],
            "configuration": json.loads(row[3]),
            "created_at": row[4],
        }

    def fit_channel_parameters(self, device_id: str) -> list[dict[str, Any]]:
        """Fit transparent noise and first-order time-constant estimates."""
        records = self.list_measurements(device_id)
        by_channel: dict[str, list[MeasurementRecord]] = {}
        for record in records:
            by_channel.setdefault(record.channel_id, []).append(record)

        fitted = []
        for channel_id, channel_records in by_channel.items():
            values = [float(record.raw_value) for record in channel_records]
            intervals = [record.sample_interval_seconds for record in channel_records]
            mean_value = sum(values) / len(values)
            differences = [b - a for a, b in zip(values, values[1:])]
            noise_floor = (
                math.sqrt(sum((value - mean_value) ** 2 for value in differences) / len(differences))
                / math.sqrt(2)
                if differences
                else 0.0
            )
            tau_seconds = self._estimate_tau(values, sum(intervals) / len(intervals))
            fitted.append(
                {
                    "device_id": device_id,
                    "channel_id": channel_id,
                    "sample_count": len(values),
                    "mean_value": mean_value,
                    "noise_floor": noise_floor,
                    "estimated_tau_seconds": tau_seconds,
                    "parameter_source": "dashboard_measurements",
                }
            )
        return fitted

    @staticmethod
    def _estimate_tau(values: list[float], sample_interval: float) -> float | None:
        if len(values) < 3:
            return None
        mean_value = sum(values) / len(values)
        centered = [value - mean_value for value in values]
        denominator = sum(value * value for value in centered[:-1])
        if denominator <= 0:
            return None
        correlation = sum(a * b for a, b in zip(centered, centered[1:])) / denominator
        if 0 < correlation < 1:
            return -sample_interval / math.log(correlation)
        return None
