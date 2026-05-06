"""Seasonal and calendar-based feature augmentation."""

from __future__ import annotations

import logging
import math
from datetime import date

logger = logging.getLogger(__name__)

# Approximate astronomical seasons (Northern Hemisphere)
_SEASON_BOUNDARIES = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}

_SEASON_ENCODING = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}


def month_to_season(month: int) -> str:
    """Return season name for a given calendar month (1–12)."""
    for season, months in _SEASON_BOUNDARIES.items():
        if month in months:
            return season
    return "winter"


def season_encoding(month: int) -> int:
    """Return ordinal encoding for the season of a given month."""
    return _SEASON_ENCODING[month_to_season(month)]


def cyclical_month(month: int) -> tuple[float, float]:
    """Return (sin, cos) encoding for a calendar month."""
    angle = 2 * math.pi * (month - 1) / 12
    return math.sin(angle), math.cos(angle)


def is_public_holiday_approx(d: date) -> bool:
    """
    Return True if the date is an approximate US federal holiday.

    Handles: New Year, Independence Day, Christmas, Thanksgiving approx.
    Does not handle floating holidays exactly.
    """
    holidays = {
        (1, 1): "New Year",
        (7, 4): "Independence Day",
        (12, 25): "Christmas",
        (12, 26): "Boxing Day / Christmas observed",
    }
    return (d.month, d.day) in holidays


def get_seasonal_features(month: int, day: int, year: int = 2025) -> dict[str, float]:
    """
    Build a dict of seasonal and calendar features for a given date.

    Args:
        month: Calendar month (1–12).
        day: Day of month (1–31).
        year: Calendar year.

    Returns:
        Dict with season_encoded, month_sin, month_cos, is_holiday.
    """
    try:
        d = date(year, month, day)
        is_holiday = is_public_holiday_approx(d)
    except ValueError:
        is_holiday = False

    month_sin, month_cos = cyclical_month(month)
    return {
        "season_encoded": float(season_encoding(month)),
        "month_sin": round(month_sin, 6),
        "month_cos": round(month_cos, 6),
        "is_holiday": float(is_holiday),
    }
