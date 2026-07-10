"""Batch prediction utilities for bulk energy consumption estimation."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.features import build_feature_matrix
from app.validators import validate_predict_input

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 1000

__all__ = ["BatchPredictRequest", "batch_predict", "MAX_BATCH_SIZE"]


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

    Builds a single vectorized DataFrame for all valid records to avoid
    per-row feature engineering overhead.

    Args:
        records: List of dicts matching PredictRequest schema.

    Returns:
        List of dicts with original input + predicted_kwh.

    Raises:
        ValueError: If batch size exceeds MAX_BATCH_SIZE.
    """
    from app.model import load_model

    if len(records) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size {len(records)} exceeds maximum {MAX_BATCH_SIZE}.")
    if not records:
        return []

    logger.info("Running batch prediction for %d records", len(records))

    rows: list[dict[str, Any]] = []
    parse_errors: list[str | None] = []
    for rec in records:
        try:
            rows.append(
                {
                    "zone": str(rec["zone"]),
                    "hour": int(rec["hour"]),
                    "day_of_week": int(rec["day_of_week"]),
                    "temperature": float(rec["temperature"]),
                    "humidity": float(rec["humidity"]),
                }
            )
            parse_errors.append(None)
        except (KeyError, TypeError, ValueError) as exc:
            rows.append({})
            parse_errors.append(str(exc))

    valid_indices = [i for i, e in enumerate(parse_errors) if e is None]
    results: list[dict[str, Any]] = [
        {**rec, "predicted_kwh": None, "error": err} for rec, err in zip(records, parse_errors, strict=False)
    ]

    if valid_indices:
        try:
            valid_rows = [rows[i] for i in valid_indices]
            raw_df = pd.DataFrame(valid_rows)
            features = build_feature_matrix(raw_df)
            ensemble = load_model()
            blend_alpha = 0.7
            lgbm_preds = ensemble["lgbm"].predict(features)
            rf_preds = ensemble["rf"].predict(features)
            blended = (blend_alpha * lgbm_preds + (1 - blend_alpha) * rf_preds).clip(min=0.0)
            for batch_pos, orig_idx in enumerate(valid_indices):
                results[orig_idx]["predicted_kwh"] = round(float(blended[batch_pos]), 4)
        except Exception as exc:
            logger.warning("Vectorized batch prediction failed, falling back: %s", exc)
            for i in valid_indices:
                results[i]["error"] = str(exc)

    successful = sum(1 for r in results if r["error"] is None)
    logger.info("Batch complete — %d/%d successful", successful, len(records))
    return results
