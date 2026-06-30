"""Input validation helpers beyond Pydantic field constraints."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ['validate_predict_input', 'validate_batch_input', 'VALID_ZONES']


VALID_ZONES = frozenset({"residential", "commercial", "industrial", "mixed"})
HOUR_RANGE = range(0, 24)
DOW_RANGE = range(0, 7)
TEMP_RANGE = (-20.0, 50.0)
HUMIDITY_RANGE = (0.0, 100.0)

REQUIRED_FIELDS = frozenset({"zone", "hour", "day_of_week", "temperature", "humidity"})


def validate_predict_input(data: dict[str, Any]) -> list[str]:
    """
    Validate a prediction input dict beyond Pydantic constraints.

    Returns:
        List of validation error messages (empty when input is valid).
    """
    errors: list[str] = []

    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}.")
        return errors

    zone = data.get("zone", "")
    if zone not in VALID_ZONES:
        errors.append(f"Invalid zone '{zone}'. Must be one of {sorted(VALID_ZONES)}.")

    hour = data.get("hour")
    if hour is not None and hour not in HOUR_RANGE:
        errors.append(f"hour must be 0–23, got {hour}.")

    dow = data.get("day_of_week")
    if dow is not None and dow not in DOW_RANGE:
        errors.append(f"day_of_week must be 0–6, got {dow}.")

    temp = data.get("temperature")
    if temp is not None and not (TEMP_RANGE[0] <= temp <= TEMP_RANGE[1]):
        errors.append(f"temperature must be {TEMP_RANGE[0]}–{TEMP_RANGE[1]} °C, got {temp}.")

    hum = data.get("humidity")
    if hum is not None and not (HUMIDITY_RANGE[0] <= hum <= HUMIDITY_RANGE[1]):
        errors.append(f"humidity must be {HUMIDITY_RANGE[0]}–{HUMIDITY_RANGE[1]} %, got {hum}.")

    if errors:
        logger.debug("Validation errors: %s", errors)
    return errors


def validate_batch_input(records: list[dict[str, Any]]) -> dict[int, list[str]]:
    """
    Validate every record in a batch.

    Returns:
        Dict mapping record index to list of error messages for that record.
        Records with no errors are omitted.
    """
    return {
        i: errs
        for i, rec in enumerate(records)
        if (errs := validate_predict_input(rec))
    }
