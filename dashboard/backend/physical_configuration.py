"""Validated per-channel physical configuration for the detector."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from measurement_contract import FrontEndType, SensorType


class JunctionStyle(StrEnum):
    EXPOSED = "exposed"
    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"
    NOT_APPLICABLE = "not_applicable"


class TauMeasurementMethod(StrEnum):
    COMMISSIONING_STEP = "commissioning_step"
    ONLINE_TRANSIENT_FIT = "online_transient_fit"
    BOTH = "commissioning_step_and_online_fit"


class SensorPair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    channel_id: str = Field(min_length=1)
    relationship: str = Field(min_length=1, pattern=r"^(fast|slow)$")
    expected_offset_seconds: float = 0.0


class ChannelPhysicalConfiguration(BaseModel):
    """All physical priors required to interpret one sensor channel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_version: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    board_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    sensor_type: SensorType
    front_end_type: FrontEndType
    junction_style: JunctionStyle
    probe_diameter_mm: float | None = Field(default=None, gt=0)
    installation_location: str = Field(min_length=1)
    insertion_depth_mm: float | None = Field(default=None, ge=0)
    thermal_time_constant_seconds: float = Field(gt=0)
    tau_uncertainty_seconds: float = Field(ge=0)
    tau_measurement_method: TauMeasurementMethod
    tau_measured_at: str = Field(min_length=1)
    minimum_physical_temperature_c: float
    maximum_physical_temperature_c: float
    maximum_source_temperature_c: float
    minimum_sink_temperature_c: float
    noise_floor_c: float = Field(ge=0)
    sample_interval_seconds: float = Field(gt=0)
    datasheet_resolution_c: float = Field(gt=0)
    thermocouple_tolerance_class: str | None = None
    related_sensor_pairs: tuple[SensorPair, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_physical_bounds(self) -> "ChannelPhysicalConfiguration":
        if self.minimum_physical_temperature_c >= self.maximum_physical_temperature_c:
            raise ValueError("minimum physical temperature must be below maximum")
        if self.minimum_sink_temperature_c > self.maximum_source_temperature_c:
            raise ValueError("minimum sink temperature must not exceed maximum source temperature")
        if self.tau_uncertainty_seconds >= self.thermal_time_constant_seconds:
            raise ValueError("tau uncertainty must be smaller than measured tau")
        if self.tau_measurement_method not in (
            TauMeasurementMethod.COMMISSIONING_STEP,
            TauMeasurementMethod.ONLINE_TRANSIENT_FIT,
            TauMeasurementMethod.BOTH,
        ):
            raise ValueError("tau must be measured during commissioning or fitted online")
        if self.front_end_type in (FrontEndType.MAX31855, FrontEndType.MAX31856) and not self.thermocouple_tolerance_class:
            raise ValueError("thermocouple tolerance class is required for thermocouple front-ends")
        return self

    def as_edge_payload(self) -> dict[str, Any]:
        """Return the versioned payload sent to an acquisition node."""
        return self.model_dump(mode="json")
