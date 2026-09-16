"""Authority rules for implausible, suspicious, and unusual quality flags."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from measurement_contract import MeasurementRecord, QualityClass, QualityFlag


STATISTICAL_DETECTORS = frozenset(
    {
        "kde_threshold",
        "reconstruction_error",
        "statistical_baseline",
        "isolation_forest",
        "autoencoder",
    }
)


class QualityPolicyError(ValueError):
    """Raised when a flag violates the authority model."""


def is_statistical_detector(detector: str) -> bool:
    normalized = detector.lower()
    return normalized in STATISTICAL_DETECTORS or any(
        token in normalized for token in ("kde", "reconstruction", "statistical", "autoencoder")
    )


def enforce_flag_authority(flags: Iterable[QualityFlag]) -> tuple[QualityFlag, ...]:
    """Ensure statistical detectors can annotate only as ``unusual``."""
    normalized: list[QualityFlag] = []
    for flag in flags:
        if is_statistical_detector(flag.detector) and flag.classification is not QualityClass.UNUSUAL:
            normalized.append(
                flag.model_copy(
                    update={
                        "classification": QualityClass.UNUSUAL,
                        "message": f"Statistical annotation: {flag.message}",
                    }
                )
            )
        else:
            normalized.append(flag)
    return tuple(normalized)


def quality_summary(record: MeasurementRecord) -> dict[str, int | bool]:
    counts = Counter(flag.classification.value for flag in record.quality_flags)
    implausible = counts[QualityClass.IMPLAUSIBLE.value]
    return {
        "implausible": implausible,
        "suspicious": counts[QualityClass.SUSPICIOUS.value],
        "unusual": counts[QualityClass.UNUSUAL.value],
        "science_excluded": implausible > 0,
        "raw_retained": True,
    }


def science_view(records: Iterable[MeasurementRecord]) -> list[MeasurementRecord]:
    """Return records eligible for science analysis without deleting raw data."""
    return [
        record
        for record in records
        if not any(
            flag.classification is QualityClass.IMPLAUSIBLE
            for flag in record.quality_flags
        )
    ]
