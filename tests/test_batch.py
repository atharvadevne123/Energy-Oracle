"""Tests for batch prediction utilities."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _train_model(tmp_path, monkeypatch):
    from app import model as m
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.model import generate_synthetic_data, train_model
    df, y = generate_synthetic_data(n_samples=200)
    train_model(df, y, cv_folds=2)


def test_batch_predict_single_record():
    from app.batch import batch_predict

    records = [{
        "zone": "residential", "hour": 8, "day_of_week": 1,
        "temperature": 20.0, "humidity": 55.0,
    }]
    results = batch_predict(records)
    assert len(results) == 1
    assert results[0]["predicted_kwh"] is not None
    assert results[0]["error"] is None


def test_batch_predict_multiple_records():
    from app.batch import batch_predict

    records = [
        {"zone": "residential", "hour": h, "day_of_week": 1, "temperature": 22.0, "humidity": 60.0}
        for h in range(5)
    ]
    results = batch_predict(records)
    assert len(results) == 5
    assert all(r["error"] is None for r in results)
    assert all(r["predicted_kwh"] > 0 for r in results)


def test_batch_predict_exceeds_max_raises():
    from app.batch import MAX_BATCH_SIZE, batch_predict

    records = [
        {"zone": "mixed", "hour": 0, "day_of_week": 0, "temperature": 15.0, "humidity": 50.0}
        for _ in range(MAX_BATCH_SIZE + 1)
    ]
    with pytest.raises(ValueError, match="Batch size"):
        batch_predict(records)


def test_batch_predict_validates_each_record():
    from app.batch import BatchPredictRequest

    records = [
        {"zone": "residential", "hour": 10, "day_of_week": 2, "temperature": 20.0, "humidity": 55.0},
        {"zone": "invalid_zone", "hour": 99, "day_of_week": 2, "temperature": 20.0, "humidity": 55.0},
    ]
    req = BatchPredictRequest(records)
    errors = req.validate()
    assert len(errors) >= 2  # zone error + hour error for record 1
