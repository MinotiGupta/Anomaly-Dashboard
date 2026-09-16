"""Deterministic, configuration-driven physics checks for sensor measurements."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from measurement_contract import MeasurementRecord, QualityClass, QualityFlag
from physical_configuration import ChannelPhysicalConfiguration


class PhysicsDetector:
    """Evaluate physics in the prescribed order after signal integrity checks."""

    def evaluate(
        self,
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
        state: dict[str, Any],
        peer_record: MeasurementRecord | None = None,
        existing_flags: list[QualityFlag] | None = None,
    ) -> list[QualityFlag]:
        flags: list[QualityFlag] = []
        flags.extend(self._bounds(record, configuration))
        flags.extend(self._slope(record, configuration, state))
        flags.extend(self._cold_soak(record, configuration))
        flags.extend(self._high_pass(record, configuration, state))
        flags.extend(self._lag(record, configuration, peer_record))
        flags.extend(self._tau_drift(record, configuration, state))
        flags.extend(
            self._gated_drift(
                record,
                configuration,
                state,
                (existing_flags or []) + flags,
            )
        )
        self._update_state(record, state, flags)
        return flags

    @staticmethod
    def _bounds(
        record: MeasurementRecord, configuration: ChannelPhysicalConfiguration
    ) -> list[QualityFlag]:
        value = float(record.raw_value)
        lower = max(
            configuration.minimum_physical_temperature_c,
            configuration.minimum_sink_temperature_c,
        )
        upper = min(
            configuration.maximum_physical_temperature_c,
            configuration.maximum_source_temperature_c,
        )
        if value < lower or value > upper:
            return [
                PhysicsDetector._flag(
                    "temperature_out_of_physical_bounds",
                    QualityClass.IMPLAUSIBLE,
                    f"Reading {value:g} C is outside [{lower:g}, {upper:g}] C",
                    "thermodynamic_bounds",
                    record,
                )
            ]
        return []

    @staticmethod
    def _slope(
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
        state: dict[str, Any],
    ) -> list[QualityFlag]:
        previous_value = state.get("physics_last_raw_value")
        previous_uptime = state.get("physics_last_uptime")
        if previous_value is None or previous_uptime is None:
            return []
        delta_seconds = record.monotonic_uptime_seconds - float(previous_uptime)
        if delta_seconds <= 0:
            return []
        slope = abs((float(record.raw_value) - float(previous_value)) / delta_seconds)
        tau = configuration.thermal_time_constant_seconds
        drive_gap = max(
            abs(configuration.maximum_source_temperature_c - float(record.raw_value)),
            abs(configuration.minimum_sink_temperature_c - float(record.raw_value)),
        )
        slope_limit = (drive_gap / max(tau, 1e-12)) * (
            1.0 + configuration.tau_uncertainty_seconds / tau
        )
        if slope > slope_limit:
            return [
                PhysicsDetector._flag(
                    "thermal_slope_limit_exceeded",
                    QualityClass.IMPLAUSIBLE,
                    f"Observed slope {slope:g} C/s exceeds physical limit {slope_limit:g} C/s",
                    "thermal_time_constant_slope",
                    record,
                )
            ]
        return []

    @staticmethod
    def _cold_soak(
        record: MeasurementRecord, configuration: ChannelPhysicalConfiguration
    ) -> list[QualityFlag]:
        metadata = record.event_metadata
        active = metadata.attributes.get("cold_soak_active", False)
        reference = metadata.attributes.get("cold_soak_reference_c")
        if not active or reference is None:
            return []
        tolerance = max(
            configuration.noise_floor_c * 3.0,
            configuration.datasheet_resolution_c * 3.0,
        )
        if abs(float(record.raw_value) - float(reference)) > tolerance:
            return [
                PhysicsDetector._flag(
                    "cold_soak_self_check_failed",
                    QualityClass.IMPLAUSIBLE,
                    "Channel does not agree with the cold-soak reference",
                    "cold_soak_self_check",
                    record,
                )
            ]
        return []

    @staticmethod
    def _high_pass(
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
        state: dict[str, Any],
    ) -> list[QualityFlag]:
        previous = state.get("physics_last_raw_value")
        if previous is None:
            return []
        residual = float(record.raw_value) - float(previous)
        residuals = list(state.get("high_pass_residuals", []))[-19:]
        residuals.append(residual)
        state["high_pass_residuals"] = residuals
        if len(residuals) < 3:
            return []
        rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        corner_frequency = 1.0 / (2.0 * math.pi * configuration.thermal_time_constant_seconds)
        cutoff_frequency = 10.0 * corner_frequency
        threshold = max(configuration.noise_floor_c * 3.0, configuration.datasheet_resolution_c * 3.0)
        spectral_energy = record.raw_register_data.get("high_pass_rms")
        observed = max(rms, float(spectral_energy)) if spectral_energy is not None else rms
        if observed > threshold and cutoff_frequency > 0:
            return [
                PhysicsDetector._flag(
                    "high_pass_energy_exceeded",
                    QualityClass.UNUSUAL,
                    f"High-pass RMS {observed:g} C exceeds noise floor {threshold:g} C above {cutoff_frequency:g} Hz",
                    "high_pass_energy",
                    record,
                )
            ]
        return []

    @staticmethod
    def _lag(
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
        peer_record: MeasurementRecord | None,
    ) -> list[QualityFlag]:
        if peer_record is None:
            peer_value = record.raw_register_data.get("peer_value")
            peer_time = record.raw_register_data.get("peer_monotonic_uptime_seconds")
            expected_offset = record.raw_register_data.get("peer_expected_offset_seconds")
        else:
            peer_value = peer_record.raw_value
            peer_time = peer_record.monotonic_uptime_seconds
            expected_offset = record.timing.cross_sensor_offset_seconds
        if peer_value is None or peer_time is None:
            return []
        offset = abs(
            record.monotonic_uptime_seconds
            - float(peer_time)
            - float(expected_offset or 0.0)
        )
        tolerance = max(configuration.sample_interval_seconds, configuration.noise_floor_c)
        if offset > tolerance:
            return [
                PhysicsDetector._flag(
                    "cross_sensor_timing_mismatch",
                    QualityClass.SUSPICIOUS,
                    "Cross-sensor sample timing differs from configured offset",
                    "two_sensor_lag",
                    record,
                )
            ]
        if abs(float(record.raw_value) - float(peer_value)) > max(
            configuration.noise_floor_c * 10.0,
            configuration.tau_uncertainty_seconds,
        ) and configuration.related_sensor_pairs:
            return [
                PhysicsDetector._flag(
                    "cross_sensor_lag_residual",
                    QualityClass.UNUSUAL,
                    "Cross-sensor residual exceeds configured comparison scale",
                    "two_sensor_lag",
                    record,
                )
            ]
        return []

    @staticmethod
    def _tau_drift(
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
        state: dict[str, Any],
    ) -> list[QualityFlag]:
        values = list(state.get("physics_values", []))[-31:]
        if len(values) < 4:
            return []
        values.append(float(record.raw_value))
        interval = record.sample_interval_seconds
        centered = [value - sum(values) / len(values) for value in values]
        denominator = sum(value * value for value in centered[:-1])
        if denominator <= 0:
            return []
        correlation = sum(a * b for a, b in zip(centered, centered[1:])) / denominator
        if not 0 < correlation < 1:
            return []
        fitted_tau = -interval / math.log(correlation)
        difference = abs(fitted_tau - configuration.thermal_time_constant_seconds)
        allowed = max(
            configuration.tau_uncertainty_seconds * 3.0,
            configuration.thermal_time_constant_seconds * 0.25,
        )
        if difference > allowed:
            return [
                PhysicsDetector._flag(
                    "thermal_time_constant_drift",
                    QualityClass.SUSPICIOUS,
                    f"Fitted tau {fitted_tau:g}s differs from configured tau {configuration.thermal_time_constant_seconds:g}s",
                    "online_tau_fit",
                    record,
                )
            ]
        return []

    @staticmethod
    def _gated_drift(
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
        state: dict[str, Any],
        flags: list[QualityFlag],
    ) -> list[QualityFlag]:
        attributes = record.event_metadata.attributes
        if not attributes.get("quiescent_window", False) or any(
            flag.classification is QualityClass.IMPLAUSIBLE for flag in flags
        ):
            return []
        previous = state.get("physics_last_raw_value")
        if previous is not None and abs(float(record.raw_value) - float(previous)) > max(
            configuration.noise_floor_c * 3.0,
            configuration.datasheet_resolution_c * 3.0,
        ):
            return []
        baseline = state.get("drift_baseline", float(record.raw_value))
        cusum = max(
            0.0,
            float(state.get("drift_cusum", 0.0))
            + float(record.raw_value)
            - float(baseline)
            - configuration.noise_floor_c,
        )
        state["drift_baseline"] = baseline
        state["drift_cusum"] = cusum
        if cusum > max(configuration.noise_floor_c * 5.0, configuration.datasheet_resolution_c * 5.0):
            return [
                PhysicsDetector._flag(
                    "gated_drift_detected",
                    QualityClass.SUSPICIOUS,
                    "Cumulative drift exceeded the quiescent-window threshold",
                    "gated_drift",
                    record,
                )
            ]
        return []

    @staticmethod
    def _update_state(
        record: MeasurementRecord, state: dict[str, Any], flags: list[QualityFlag]
    ) -> None:
        state["physics_last_raw_value"] = float(record.raw_value)
        state["physics_last_uptime"] = record.monotonic_uptime_seconds
        state["physics_values"] = list(state.get("physics_values", []))[-31:] + [
            float(record.raw_value)
        ]
        state["physics_last_flag_codes"] = [flag.code for flag in flags]

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
            detected_at=datetime.now(timezone.utc),
        )
