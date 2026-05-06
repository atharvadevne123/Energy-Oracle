"""Zone consumption profiles and statistical baseline utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZoneProfile:
    """Statistical profile for an energy consumption zone."""

    name: str
    baseline_kwh: float
    peak_multiplier: float
    off_peak_multiplier: float
    weekend_adjustment: float
    typical_temp_sensitivity: float

    # Zone registry
    _registry: ClassVar[dict[str, ZoneProfile]] = {}

    def __post_init__(self) -> None:
        ZoneProfile._registry[self.name] = self

    def expected_kwh(self, hour: int, is_weekend: bool, temperature: float) -> float:
        """Estimate expected kWh for given conditions."""
        if 17 <= hour <= 21:
            base = self.baseline_kwh * self.peak_multiplier
        elif 0 <= hour <= 5:
            base = self.baseline_kwh * self.off_peak_multiplier
        else:
            base = self.baseline_kwh

        if is_weekend:
            base *= (1 + self.weekend_adjustment)

        temp_delta = temperature - 18.0
        base += temp_delta * self.typical_temp_sensitivity
        return max(0.0, round(base, 2))


RESIDENTIAL = ZoneProfile(
    name="residential",
    baseline_kwh=25.0,
    peak_multiplier=1.4,
    off_peak_multiplier=0.5,
    weekend_adjustment=0.1,
    typical_temp_sensitivity=0.8,
)

COMMERCIAL = ZoneProfile(
    name="commercial",
    baseline_kwh=80.0,
    peak_multiplier=1.3,
    off_peak_multiplier=0.3,
    weekend_adjustment=-0.3,
    typical_temp_sensitivity=2.0,
)

INDUSTRIAL = ZoneProfile(
    name="industrial",
    baseline_kwh=250.0,
    peak_multiplier=1.2,
    off_peak_multiplier=0.7,
    weekend_adjustment=-0.2,
    typical_temp_sensitivity=5.0,
)

MIXED = ZoneProfile(
    name="mixed",
    baseline_kwh=60.0,
    peak_multiplier=1.35,
    off_peak_multiplier=0.4,
    weekend_adjustment=-0.05,
    typical_temp_sensitivity=1.5,
)

ALL_ZONES: dict[str, ZoneProfile] = ZoneProfile._registry


def get_zone_profile(zone: str) -> ZoneProfile:
    """Return zone profile by name, defaulting to mixed."""
    if zone not in ALL_ZONES:
        logger.warning("Unknown zone '%s' — falling back to mixed", zone)
        return MIXED
    return ALL_ZONES[zone]
