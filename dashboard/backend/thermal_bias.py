"""Expected thermal and instrumentation bias annotations."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from measurement_contract import (
    DiagnosticAnnotation,
    FrontEndType,
    MeasurementRecord,
)
from physical_configuration import ChannelPhysicalConfiguration, JunctionStyle


class ThermalBiasAnalyzer:
    """Annotate expected measurement effects without raising quality flags."""

    STEFAN_BOLTZMANN = 5.670374419e-8

    def analyze(
        self,
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
        state: dict[str, Any],
    ) -> list[DiagnosticAnnotation]:
        annotations: list[DiagnosticAnnotation] = []
        annotations.extend(self._radiation(record, configuration))
        annotations.extend(self._conduction(record, configuration))
        annotations.extend(self._recovery(record, configuration))
        annotations.extend(self._inhomogeneity(record))
        annotations.extend(self._cold_junction(record, state))
        annotations.extend(self._linearisation(record))
        annotations.extend(self._junction_style(record, configuration))
        self._update_state(record, state)
        return annotations

    def _radiation(
        self,
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
    ) -> list[DiagnosticAnnotation]:
        data = record.raw_register_data
        wall = data.get("wall_temperature_c")
        heat_transfer = data.get("heat_transfer_coefficient_w_m2k")
        emissivity = data.get("emissivity", configuration.metadata.get("emissivity"))
        if wall is None or heat_transfer is None or emissivity is None:
            return []
        kelvin = float(record.raw_value) + 273.15
        wall_kelvin = float(wall) + 273.15
        h = float(heat_transfer)
        if h <= 0:
            return []
        bias_kelvin = float(emissivity) * self.STEFAN_BOLTZMANN * (
            kelvin**4 - wall_kelvin**4
        ) / h
        return [
            self._annotation(
                "radiation_error_expected",
                f"Radiation exchange may bias the thermocouple by about {bias_kelvin:g} C; retain the reading",
                bias_kelvin,
                {"wall_temperature_c": wall, "heat_transfer_coefficient_w_m2k": h, "emissivity": emissivity},
            )
        ]

    def _conduction(
        self,
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
    ) -> list[DiagnosticAnnotation]:
        data = record.raw_register_data
        mount = data.get("mount_temperature_c")
        if mount is None or configuration.insertion_depth_mm is None or configuration.probe_diameter_mm is None:
            return []
        ratio = configuration.insertion_depth_mm / configuration.probe_diameter_mm
        if ratio >= 20:
            return []
        attenuation = math.exp(-ratio / 10.0)
        bias = (float(mount) - float(record.raw_value)) * attenuation
        return [
            self._annotation(
                "conduction_error_expected",
                "Probe stem conduction may bias the reading toward the mount temperature",
                bias,
                {"mount_temperature_c": mount, "depth_to_diameter_ratio": ratio},
            )
        ]

    def _recovery(
        self,
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
    ) -> list[DiagnosticAnnotation]:
        data = record.raw_register_data
        velocity = data.get("gas_velocity_m_s")
        if velocity is None or float(velocity) <= 50.0:
            return []
        recovery_factor = float(data.get("recovery_factor", configuration.metadata.get("recovery_factor", 0.77)))
        heat_capacity = float(data.get("gas_specific_heat_j_kgk", 1200.0))
        bias = recovery_factor * float(velocity) ** 2 / (2.0 * heat_capacity)
        return [
            self._annotation(
                "recovery_effect_expected",
                "High gas velocity can raise the probe reading through recovery heating",
                bias,
                {"gas_velocity_m_s": velocity, "recovery_factor": recovery_factor},
            )
        ]

    def _inhomogeneity(self, record: MeasurementRecord) -> list[DiagnosticAnnotation]:
        data = record.raw_register_data
        if not any(data.get(key) is True for key in ("wire_repositioned", "inhomogeneity_signature", "temperature_band_error")):
            return []
        return [
            self._annotation(
                "thermocouple_inhomogeneity_diagnostic",
                "Thermocouple wire inhomogeneity may cause position- or temperature-band-dependent bias; software calibration cannot remove it",
                None,
                {key: data[key] for key in ("wire_repositioned", "inhomogeneity_signature", "temperature_band_error") if key in data},
            )
        ]

    def _cold_junction(self, record: MeasurementRecord, state: dict[str, Any]) -> list[DiagnosticAnnotation]:
        board_temperature = record.raw_register_data.get("board_temperature_c")
        if board_temperature is None:
            return []
        previous = state.get("last_board_temperature_c")
        state["last_board_temperature_c"] = float(board_temperature)
        if previous is None or abs(float(board_temperature) - previous) < 0.5:
            return []
        return [
            self._annotation(
                "cold_junction_drift_diagnostic",
                "Board-temperature movement may create a common-mode cold-junction compensation shift",
                None,
                {"previous_board_temperature_c": previous, "board_temperature_c": board_temperature},
            )
        ]

    def _linearisation(self, record: MeasurementRecord) -> list[DiagnosticAnnotation]:
        data = record.raw_register_data
        if record.front_end_type is not FrontEndType.MAX31855:
            return []
        expected_difference = data.get("max31855_linearisation_difference_c")
        if expected_difference is None:
            return []
        return [
            self._annotation(
                "max31855_linearisation_difference_expected",
                "MAX31855 constant-slope linearisation differs from ITS-90/reference behaviour; do not classify this known bias as a fault",
                float(expected_difference),
                {"reference_front_end": data.get("reference_front_end", "MAX31856_or_DAQ970A")},
            )
        ]

    def _junction_style(
        self,
        record: MeasurementRecord,
        configuration: ChannelPhysicalConfiguration,
    ) -> list[DiagnosticAnnotation]:
        if configuration.junction_style is JunctionStyle.NOT_APPLICABLE:
            return []
        return [
            self._annotation(
                "junction_style_diagnostic",
                f"{configuration.junction_style.value} junction changes thermal response and electrical susceptibility; interpret tau and noise with this configuration",
                None,
                {"junction_style": configuration.junction_style.value},
            )
        ]

    @staticmethod
    def _update_state(record: MeasurementRecord, state: dict[str, Any]) -> None:
        state["last_thermal_bias_value"] = float(record.raw_value)

    @staticmethod
    def _annotation(
        code: str,
        message: str,
        estimated_bias_c: float | None,
        evidence: dict[str, Any],
    ) -> DiagnosticAnnotation:
        return DiagnosticAnnotation(
            code=code,
            message=message,
            detector="thermal_bias_analyzer",
            estimated_bias_c=estimated_bias_c,
            evidence=evidence,
            annotated_at=datetime.now(timezone.utc),
        )
