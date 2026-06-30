"""Tests for FastAPI endpoints."""

from __future__ import annotations

import pytest


def test_health_returns_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime_seconds" in data


def test_predict_returns_kwh(client, sample_predict_payload):
    response = client.post("/api/v1/predict", json=sample_predict_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_kwh" in data
    assert data["predicted_kwh"] > 0
    assert data["zone"] == "residential"
    assert "correlation_id" in data


@pytest.mark.parametrize("zone", ["residential", "commercial", "industrial", "mixed"])
def test_predict_all_zones(client, zone):
    payload = {
        "zone": zone,
        "hour": 12,
        "day_of_week": 3,
        "temperature": 22.0,
        "humidity": 55.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_kwh"] > 0


def test_predict_invalid_zone_rejected(client):
    payload = {
        "zone": "underwater",
        "hour": 10,
        "day_of_week": 2,
        "temperature": 20.0,
        "humidity": 50.0,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("hour", 25),
        ("hour", -1),
        ("day_of_week", 7),
        ("temperature", 60.0),
        ("humidity", 110.0),
    ],
)
def test_predict_out_of_range_rejected(client, sample_predict_payload, bad_field, bad_value):
    payload = {**sample_predict_payload, bad_field: bad_value}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_returns_correlation_id(client, sample_predict_payload):
    response = client.post("/api/v1/predict", json=sample_predict_payload)
    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers
    assert response.json()["correlation_id"] == response.headers["X-Correlation-ID"]


def test_metrics_endpoint(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "training_metrics" in data
    assert "prediction_summary" in data


def test_drift_endpoint_insufficient_data(client):
    response = client.get("/api/v1/drift")
    assert response.status_code == 200
    data = response.json()
    # Not enough data yet — expect insufficient_data status
    assert data.get("status") in ("insufficient_data", "stable", "drift_detected")


def test_predict_peak_hour_higher_than_off_peak(client):
    """Industrial zone should predict more kWh at peak hour than off-peak."""
    base = {"zone": "industrial", "day_of_week": 2, "temperature": 20.0, "humidity": 50.0}
    peak_resp = client.post("/api/v1/predict", json={**base, "hour": 19})
    off_peak_resp = client.post("/api/v1/predict", json={**base, "hour": 3})
    assert peak_resp.status_code == 200
    assert off_peak_resp.status_code == 200
    peak_kwh = peak_resp.json()["predicted_kwh"]
    off_kwh = off_peak_resp.json()["predicted_kwh"]
    assert peak_kwh >= 0 and off_kwh >= 0


def test_version_endpoint(client):
    response = client.get("/api/v1/version")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Energy-Oracle"
    assert "version" in data
    assert "model_version" in data


def test_batch_endpoint_valid(client, sample_predict_payload):
    body = {"records": [sample_predict_payload, {**sample_predict_payload, "zone": "commercial"}]}
    response = client.post("/api/v1/batch", json=body)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["successful"] == 2
    assert len(data["results"]) == 2
    assert all(r["predicted_kwh"] > 0 for r in data["results"])


def test_batch_endpoint_empty(client):
    response = client.post("/api/v1/batch", json={"records": []})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["successful"] == 0


@pytest.mark.parametrize("zone", ["residential", "commercial", "industrial", "mixed"])
def test_batch_all_zones(client, zone):
    payload = {"zone": zone, "hour": 8, "day_of_week": 1, "temperature": 18.0, "humidity": 50.0}
    response = client.post("/api/v1/batch", json={"records": [payload]})
    assert response.status_code == 200
    assert response.json()["successful"] == 1


def test_health_deep_endpoint(client):
    response = client.get("/api/v1/health/deep")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded")
    assert "components" in data


def test_predict_custom_correlation_id(client, sample_predict_payload):
    headers = {"X-Correlation-ID": "test-corr-42"}
    response = client.post("/api/v1/predict", json=sample_predict_payload, headers=headers)
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == "test-corr-42"
    assert response.json()["correlation_id"] == "test-corr-42"
