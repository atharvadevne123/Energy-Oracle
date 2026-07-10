"""Tests for drift detection and monitoring utilities."""

from __future__ import annotations

import numpy as np
import pytest


def test_no_drift_on_same_distribution(reference_series, current_series_stable):
    from app.monitoring import compute_drift

    result = compute_drift(reference_series, current_series_stable)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert not result["drift_detected"], "Should not detect drift on stable distribution"


def test_drift_detected_on_shifted_distribution(reference_series, current_series_drifted):
    from app.monitoring import compute_drift

    result = compute_drift(reference_series, current_series_drifted)
    assert result["drift_detected"], "Should detect drift on shifted distribution"
    assert result["ks_statistic"] > 0.2


def test_insufficient_samples_returns_no_drift():
    from app.monitoring import compute_drift

    result = compute_drift([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert not result["drift_detected"]
    assert result.get("reason") == "insufficient_samples"


@pytest.mark.parametrize("n_ref,n_cur", [(200, 50), (500, 100)])
def test_ks_statistic_range(n_ref, n_cur):
    from app.monitoring import compute_drift

    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, n_ref).tolist()
    cur = rng.normal(0, 1, n_cur).tolist()
    result = compute_drift(ref, cur)
    assert 0 <= result["ks_statistic"] <= 1
    assert 0 <= result["p_value"] <= 1


def test_log_and_retrieve_prediction(db_session):
    import uuid

    from app.monitoring import get_recent_predictions, log_prediction

    corr_id = str(uuid.uuid4())
    log_prediction(
        db_session,
        correlation_id=corr_id,
        zone="commercial",
        hour=10,
        day_of_week=2,
        temperature=22.0,
        humidity=55.0,
        predicted_kwh=85.3,
    )

    recent = get_recent_predictions(db_session, limit=10)
    assert len(recent) >= 1
    assert any(r.correlation_id == corr_id for r in recent)


def test_summarise_predictions_empty(db_session):
    from app.monitoring import summarise_predictions

    # Fresh session may have no records
    result = summarise_predictions(db_session, limit=1)
    assert "count" in result


def test_run_drift_check_insufficient(db_session):
    from app.monitoring import run_drift_check

    result = run_drift_check(db_session)
    assert result["status"] in ("insufficient_data", "stable", "drift_detected")
