"""Tests for Pydantic API schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictRequest,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)


@pytest.mark.parametrize("zone", ["residential", "commercial", "industrial", "mixed"])
def test_predict_request_valid_zones(zone):
    req = PredictRequest(zone=zone, hour=12, day_of_week=3, temperature=20.0, humidity=55.0)
    assert req.zone == zone


def test_predict_request_zone_normalised_lowercase():
    req = PredictRequest(zone="RESIDENTIAL", hour=0, day_of_week=0, temperature=10.0, humidity=40.0)
    assert req.zone == "residential"


def test_predict_request_zone_normalised_strip():
    req = PredictRequest(
        zone="  commercial  ", hour=0, day_of_week=0, temperature=10.0, humidity=40.0
    )
    assert req.zone == "commercial"


def test_predict_request_invalid_zone():
    with pytest.raises(ValidationError):
        PredictRequest(zone="rooftop", hour=12, day_of_week=3, temperature=20.0, humidity=55.0)


@pytest.mark.parametrize("hour", [-1, 24, 100])
def test_predict_request_invalid_hour(hour):
    with pytest.raises(ValidationError):
        PredictRequest(
            zone="residential", hour=hour, day_of_week=3, temperature=20.0, humidity=55.0
        )


@pytest.mark.parametrize("dow", [-1, 7, 10])
def test_predict_request_invalid_day_of_week(dow):
    with pytest.raises(ValidationError):
        PredictRequest(
            zone="residential", hour=12, day_of_week=dow, temperature=20.0, humidity=55.0
        )


@pytest.mark.parametrize("temp", [-25.0, 55.0])
def test_predict_request_invalid_temperature(temp):
    with pytest.raises(ValidationError):
        PredictRequest(zone="residential", hour=12, day_of_week=3, temperature=temp, humidity=55.0)


@pytest.mark.parametrize("hum", [-1.0, 101.0])
def test_predict_request_invalid_humidity(hum):
    with pytest.raises(ValidationError):
        PredictRequest(zone="residential", hour=12, day_of_week=3, temperature=20.0, humidity=hum)


def test_predict_response_fields():
    resp = PredictResponse(
        predicted_kwh=42.5,
        zone="residential",
        hour=18,
        day_of_week=1,
        model_version="1.0.0",
        correlation_id="abc-123",
    )
    assert resp.predicted_kwh == 42.5
    assert resp.correlation_id == "abc-123"


def test_health_response():
    r = HealthResponse(status="ok", version="1.0.0", uptime_seconds=3.14)
    assert r.status == "ok"


def test_metrics_response():
    r = MetricsResponse(training_metrics={"rmse": 5.0}, prediction_summary={"count": 10})
    assert r.training_metrics["rmse"] == 5.0


def test_version_response():
    r = VersionResponse(name="Energy-Oracle", version="1.0.0", model_version="1.0.0")
    assert r.name == "Energy-Oracle"


def test_batch_predict_request_valid():
    req = BatchPredictRequest(
        records=[
            {
                "zone": "residential",
                "hour": 10,
                "day_of_week": 1,
                "temperature": 22.0,
                "humidity": 55.0,
            },
            {
                "zone": "commercial",
                "hour": 14,
                "day_of_week": 3,
                "temperature": 25.0,
                "humidity": 60.0,
            },
        ]
    )
    assert len(req.records) == 2


def test_batch_predict_request_empty_records():
    req = BatchPredictRequest(records=[])
    assert req.records == []
