"""Anomaly detection for energy consumption predictions using Isolation Forest."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

__all__ = ['fit_anomaly_detector', 'score_prediction', 'generate_reference_samples']



# Contamination rate — expected fraction of anomalies in training data
DEFAULT_CONTAMINATION = 0.05
_anomaly_detector: IsolationForest | None = None
_reference_data: list[list[float]] = []


def fit_anomaly_detector(
    samples: list[list[float]],
    contamination: float = DEFAULT_CONTAMINATION,
) -> IsolationForest:
    """
    Fit an Isolation Forest on reference samples.

    Args:
        samples: List of feature vectors (each vector: [temperature, humidity, kwh]).
        contamination: Expected proportion of outliers.

    Returns:
        Fitted IsolationForest model.
    """
    global _anomaly_detector, _reference_data
    _reference_data = samples
    detector = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    detector.fit(samples)
    _anomaly_detector = detector
    logger.info("Anomaly detector fitted on %d samples", len(samples))
    return detector


def score_prediction(
    temperature: float,
    humidity: float,
    predicted_kwh: float,
) -> dict[str, Any]:
    """
    Score a single prediction for anomalousness.

    Args:
        temperature: Input temperature.
        humidity: Input humidity.
        predicted_kwh: Model output.

    Returns:
        Dict with anomaly_score, is_anomaly, and detector_ready flags.
    """
    if _anomaly_detector is None:
        return {"anomaly_score": 0.0, "is_anomaly": False, "detector_ready": False}

    features = np.array([[temperature, humidity, predicted_kwh]])
    score = float(_anomaly_detector.score_samples(features)[0])
    prediction = int(_anomaly_detector.predict(features)[0])
    is_anomaly = prediction == -1

    if is_anomaly:
        logger.warning(
            "Anomaly detected — temp=%.1f hum=%.1f kwh=%.2f score=%.4f",
            temperature, humidity, predicted_kwh, score,
        )

    return {
        "anomaly_score": round(score, 4),
        "is_anomaly": is_anomaly,
        "detector_ready": True,
    }


def generate_reference_samples(n: int = 500) -> list[list[float]]:
    """Generate synthetic normal-range samples for detector initialisation."""
    rng = np.random.default_rng(42)
    temps = rng.normal(20, 8, n).clip(-5, 40)
    humidity = rng.uniform(30, 90, n)
    kwh = rng.uniform(10, 300, n)
    return [[float(t), float(h), float(k)] for t, h, k in zip(temps, humidity, kwh, strict=False)]
