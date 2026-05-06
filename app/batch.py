"""Batch prediction utilities for bulk energy consumption estimation."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.features import build_feature_matrix
from app.validators import validate_predict_input

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 1000


class BatchPredictRequest:
    """Container for a batch prediction job."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def validate(self) -> list[str]:
        """Validate all records and return a flat list of errors."""
        errors: list[str] = []
        for i, rec in enumerate(self.records):
            rec_errors = validate_predict_input(rec)
            for e in rec_errors:
                errors.append(f"Record {i}: {e}")
        return errors


def batch_predict(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Run predictions on a list of input records.

    Args:
        records: List of dicts matching PredictRequest schema.

    Returns:
        List of dicts with original input + predicted_kwh.

    Raises:
        ValueError: If batch size exceeds MAX_BATCH_SIZE.
    """
    from app.model import predict

    if len(records) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(records)} exceeds maximum {MAX_BATCH_SIZE}.")

    logger.info("Running batch prediction for %d records", len(records))
    results = []

    for rec in records:
        try:
            raw_df = pd.DataFrame([{
                "zone": rec["zone"],
                "hour": int(rec["hour"]),
                "day_of_week": int(rec["day_of_week"]),
                "temperature": float(rec["temperature"]),
                "humidity": float(rec["humidity"]),
            }])
            features = build_feature_matrix(raw_df)
            kwh = predict(features)
            results.append({**rec, "predicted_kwh": kwh, "error": None})
        except Exception as exc:
            logger.warning("Batch record error: %s", exc)
            results.append({**rec, "predicted_kwh": None, "error": str(exc)})

    successful = sum(1 for r in results if r["error"] is None)
    logger.info("Batch complete — %d/%d successful", successful, len(records))
    return results
