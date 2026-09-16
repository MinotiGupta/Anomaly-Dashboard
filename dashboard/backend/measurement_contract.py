"""Versioned raw measurement contract for the anomaly detection pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


MEASUREMENT_SCHEMA_VERSION = "1.0"


class SensorType(StrEnum):
    K_TYPE_THERMOCOUPLE = "k_type_thermocouple"
    DS18B20 = "ds18b20"
    SHT45 = "sht45"
    DHT11 = "dht11"
    UNKNOWN = "unknown"


class FrontEndType(StrEnum):
    MAX31855 = "max31855"
    MAX31856 = "max31856"
    DS18B20_1WIRE = "ds18b20_1wire"
    DAQ970A = "daq970a"
    UNKNOWN = "unknown"


class QualityClass(StrEnum):
    IMPLAUSIBLE = "implausible"
    SUSPICIOUS = "suspicious"
    UNUSUAL = "unusual"


class QualityFlag(BaseModel):
    """An additive detector result; it never replaces the raw measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    classification: QualityClass
    message: str = Field(min_length=1)
    detector: str = Field(min_length=1)
    detected_at: datetime

    @field_validator("detected_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("detected_at must include timezone information")
        return value


class EventMetadata(BaseModel):
    """Process state captured alongside a measurement when available."""

    model_config = ConfigDict(frozen=True, extra="allow")

    igniter_state: str | None = None
    valve_state: str | None = None
    pump_state: str | None = None
    shutdown_state: str | None = None
    event_id: str | None = None
    attributes: Mapping[str, Any] = Field(default_factory=dict)


class MeasurementRecord(BaseModel):
    """One raw sensor sample plus immutable context and additive quality flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default=MEASUREMENT_SCHEMA_VERSION, min_length=1)
    measurement_id: str = Field(min_length=1)
    raw_value: float | int
    wall_clock_timestamp: datetime
    monotonic_uptime_seconds: float = Field(ge=0)
    boot_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    sensor_type: SensorType
    front_end_type: FrontEndType
    sample_interval_seconds: float = Field(gt=0)
    raw_hardware_word: int | None = None
    raw_register_data: Mapping[str, int | float | str | bool | None] = Field(
        default_factory=dict
    )
    quality_flags: tuple[QualityFlag, ...] = ()
    detector_ruleset_version: str = Field(min_length=1)
    parameter_set_version: str = Field(min_length=1)
    event_metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("wall_clock_timestamp")
    @classmethod
    def require_aware_wall_clock_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("wall_clock_timestamp must include timezone information")
        return value

    @field_validator("quality_flags")
    @classmethod
    def reject_duplicate_quality_codes(
        cls, value: tuple[QualityFlag, ...]
    ) -> tuple[QualityFlag, ...]:
        codes = [flag.code for flag in value]
        if len(codes) != len(set(codes)):
            raise ValueError("quality_flags cannot contain duplicate codes")
        return value

    def with_quality_flags(self, *flags: QualityFlag) -> Self:
        """Return a new record with flags appended; raw fields remain unchanged."""
        return self.model_copy(update={"quality_flags": self.quality_flags + tuple(flags)})

    def raw_payload(self) -> dict[str, Any]:
        """Serialize the complete record for transport or durable storage."""
        return self.model_dump(mode="json")
