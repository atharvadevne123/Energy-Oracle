"""Tests for Isolation Forest anomaly detection."""

from __future__ import annotations

import pytest


@pytest.fixture
def fitted_detector():
    from app.anomaly import fit_anomaly_detector, generate_reference_samples

    samples = generate_reference_samples(n=300)
    return fit_anomaly_detector(samples)


def test_score_before_fitting_returns_not_ready():
    import app.anomaly as anomaly_mod

    # Reset detector
    anomaly_mod._anomaly_detector = None
    from app.anomaly import score_prediction

    result = score_prediction(20.0, 55.0, 80.0)
    assert not result["detector_ready"]
    assert not result["is_anomaly"]


def test_normal_prediction_not_anomalous(fitted_detector):
    from app.anomaly import score_prediction

    result = score_prediction(temperature=20.0, humidity=60.0, predicted_kwh=85.0)
    assert result["detector_ready"]
    assert "anomaly_score" in result
    assert isinstance(result["is_anomaly"], bool)


def test_extreme_prediction_is_anomalous(fitted_detector):
    from app.anomaly import score_prediction

    # Extremely high kWh that's far outside the training range
    result = score_prediction(temperature=45.0, humidity=99.0, predicted_kwh=9999.0)
    assert result["detector_ready"]
    assert result["is_anomaly"]


def test_generate_reference_samples_shape():
    from app.anomaly import generate_reference_samples

    samples = generate_reference_samples(n=100)
    assert len(samples) == 100
    assert all(len(s) == 3 for s in samples)


def test_fit_returns_isolation_forest(fitted_detector):
    from sklearn.ensemble import IsolationForest

    assert isinstance(fitted_detector, IsolationForest)
