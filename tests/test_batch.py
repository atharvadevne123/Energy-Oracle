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


def test_batch_predict_empty_returns_empty():
    from app.batch import batch_predict

    results = batch_predict([])
    assert results == []


def test_batch_predict_missing_key_produces_error():
    from app.batch import batch_predict

    records = [{"zone": "residential", "hour": 10, "day_of_week": 2, "temperature": 20.0}]
    results = batch_predict(records)
    assert len(results) == 1
    assert results[0]["error"] is not None
    assert results[0]["predicted_kwh"] is None


def test_batch_predict_partial_failure():
    """Valid records produce predictions; invalid records carry error strings."""
    from app.batch import batch_predict

    records = [
        {"zone": "commercial", "hour": 9, "day_of_week": 3, "temperature": 21.0, "humidity": 58.0},
        {"zone": "industrial"},  # missing required keys
        {"zone": "mixed", "hour": 14, "day_of_week": 4, "temperature": 18.0, "humidity": 45.0},
    ]
    results = batch_predict(records)
    assert len(results) == 3
    assert results[0]["error"] is None
    assert results[1]["error"] is not None
    assert results[2]["error"] is None
    assert results[0]["predicted_kwh"] is not None
    assert results[2]["predicted_kwh"] is not None


def test_batch_predict_all_zones():
    from app.batch import batch_predict

    records = [
        {"zone": z, "hour": 12, "day_of_week": 2, "temperature": 20.0, "humidity": 50.0}
        for z in ("residential", "commercial", "industrial", "mixed")
    ]
    results = batch_predict(records)
    assert len(results) == 4
    assert all(r["error"] is None for r in results)
    assert all(r["predicted_kwh"] > 0 for r in results)


def test_batch_predict_result_order_preserved():
    """Result indices must correspond to input record order."""
    from app.batch import batch_predict

    records = [
        {"zone": "residential", "hour": 6, "day_of_week": 0, "temperature": 15.0, "humidity": 40.0},
        {"zone": "commercial", "hour": 12, "day_of_week": 3, "temperature": 25.0, "humidity": 65.0},
        {"zone": "industrial", "hour": 20, "day_of_week": 5, "temperature": 30.0, "humidity": 70.0},
    ]
    results = batch_predict(records)
    for i, (rec, res) in enumerate(zip(records, results)):
        assert res["zone"] == rec["zone"], f"Result {i} zone mismatch"


def test_batch_predict_request_validate_no_errors_on_valid():
    from app.batch import BatchPredictRequest

    records = [
        {"zone": "residential", "hour": 8, "day_of_week": 1, "temperature": 20.0, "humidity": 55.0},
    ]
    req = BatchPredictRequest(records)
    errors = req.validate()
    assert errors == []
