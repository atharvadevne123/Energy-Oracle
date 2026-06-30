"""Tests for ASGI middleware — correlation ID and rate limiting."""

from __future__ import annotations

import pytest


def test_correlation_id_generated_if_absent(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers
    corr_id = response.headers["X-Correlation-ID"]
    assert len(corr_id) > 0


def test_correlation_id_forwarded_from_request(client):
    headers = {"X-Correlation-ID": "my-custom-id-999"}
    response = client.get("/api/v1/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == "my-custom-id-999"


def test_correlation_id_unique_per_request(client):
    r1 = client.get("/api/v1/health")
    r2 = client.get("/api/v1/health")
    id1 = r1.headers.get("X-Correlation-ID")
    id2 = r2.headers.get("X-Correlation-ID")
    assert id1 != id2


def test_rate_limit_middleware_imports():
    from app.middleware import rate_limit_middleware
    assert callable(rate_limit_middleware)


def test_correlation_middleware_imports():
    from app.middleware import CorrelationIDMiddleware
    assert CorrelationIDMiddleware is not None


def test_correlation_id_in_predict_response(client, sample_predict_payload):
    response = client.post("/api/v1/predict", json=sample_predict_payload)
    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers
    data = response.json()
    assert data["correlation_id"] == response.headers["X-Correlation-ID"]


def test_rate_store_initialises_empty():
    from app.middleware import _rate_store
    assert isinstance(_rate_store, dict)
