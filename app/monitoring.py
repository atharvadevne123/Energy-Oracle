"""Model monitoring: drift detection via KS-test and prediction logging."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftEvent, PredictionLog

logger = logging.getLogger(__name__)

# Minimum window size required for a meaningful KS-test
__all__ = ["compute_drift", "log_prediction", "get_recent_predictions", "run_drift_check", "summarise_predictions", "MIN_WINDOW_FOR_DRIFT"]

MIN_WINDOW_FOR_DRIFT = 30


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """
    Run KS-test between reference and current distributions.

    Returns ks_statistic, p_value, and drift_detected flag.
    p < 0.05 is treated as statistically significant drift.
    """
    if len(reference) < MIN_WINDOW_FOR_DRIFT or len(current) < MIN_WINDOW_FOR_DRIFT:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "reason": "insufficient_samples",
        }
    stat, p = ks_2samp(reference, current)
    result = {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }
    if result["drift_detected"]:
        logger.warning(
            "Drift detected — KS=%.4f p=%.4f", result["ks_statistic"], result["p_value"]
        )
    return result


def log_prediction(
    db: Session,
    *,
    correlation_id: str,
    zone: str,
    hour: int,
    day_of_week: int,
    temperature: float,
    humidity: float,
    predicted_kwh: float,
    model_version: str = "1.0.0",
) -> PredictionLog:
    """Persist a prediction record to the database."""
    record = PredictionLog(
        correlation_id=correlation_id,
        zone=zone,
        hour=hour,
        day_of_week=day_of_week,
        temperature=temperature,
        humidity=humidity,
        predicted_kwh=predicted_kwh,
        model_version=model_version,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug("Logged prediction id=%d corr=%s", record.id, correlation_id)
    return record


def get_recent_predictions(db: Session, limit: int = 500) -> list[PredictionLog]:
    """Fetch the most recent prediction records for drift analysis."""
    return (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )


def run_drift_check(db: Session, reference_window: int = 200, current_window: int = 50) -> dict[str, Any]:
    """
    Compare recent predictions against a reference window.

    Runs KS-tests on temperature, humidity, and predicted_kwh distributions.
    """
    recent = get_recent_predictions(db, limit=reference_window + current_window)
    if len(recent) < MIN_WINDOW_FOR_DRIFT * 2:
        return {"status": "insufficient_data", "n_records": len(recent)}

    ref = recent[current_window:]
    cur = recent[:current_window]

    features_to_check = {
        "temperature": ([r.temperature for r in ref], [r.temperature for r in cur]),
        "humidity": ([r.humidity for r in ref], [r.humidity for r in cur]),
        "predicted_kwh": ([r.predicted_kwh for r in ref], [r.predicted_kwh for r in cur]),
    }

    results: dict[str, Any] = {}
    any_drift = False
    for feat, (ref_vals, cur_vals) in features_to_check.items():
        drift = compute_drift(ref_vals, cur_vals)
        results[feat] = drift
        if drift["drift_detected"]:
            any_drift = True
            event = DriftEvent(
                feature=feat,
                ks_statistic=drift["ks_statistic"],
                p_value=drift["p_value"],
                drift_detected=1,
                window_size=current_window,
            )
            db.add(event)

    db.commit()
    return {
        "status": "drift_detected" if any_drift else "stable",
        "features": results,
        "reference_window": len(ref),
        "current_window": len(cur),
    }


def summarise_predictions(db: Session, limit: int = 100) -> dict[str, Any]:
    """Return summary statistics over recent predictions."""
    records = get_recent_predictions(db, limit=limit)
    if not records:
        return {"count": 0}
    kwh_vals = [r.predicted_kwh for r in records]
    zone_counts: dict[str, int] = {}
    for r in records:
        zone_counts[r.zone] = zone_counts.get(r.zone, 0) + 1
    return {
        "count": len(records),
        "kwh_mean": round(float(np.mean(kwh_vals)), 4),
        "kwh_std": round(float(np.std(kwh_vals)), 4),
        "kwh_min": round(float(np.min(kwh_vals)), 4),
        "kwh_max": round(float(np.max(kwh_vals)), 4),
        "kwh_p50": round(float(np.percentile(kwh_vals, 50)), 4),
        "kwh_p95": round(float(np.percentile(kwh_vals, 95)), 4),
        "zones": list({r.zone for r in records}),
        "zone_counts": zone_counts,
    }
