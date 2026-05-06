"""Tests for input validation helpers."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("zone", ["residential", "commercial", "industrial", "mixed"])
def test_valid_zones_pass(zone):
    from app.validators import validate_predict_input

    errors = validate_predict_input({
        "zone": zone, "hour": 12, "day_of_week": 3,
        "temperature": 20.0, "humidity": 55.0,
    })
    assert errors == []


@pytest.mark.parametrize("bad_zone", ["office", "factory", "", "RESIDENTIAL"])
def test_invalid_zones_fail(bad_zone):
    from app.validators import validate_predict_input

    errors = validate_predict_input({
        "zone": bad_zone, "hour": 12, "day_of_week": 3,
        "temperature": 20.0, "humidity": 55.0,
    })
    assert any("zone" in e.lower() for e in errors)


@pytest.mark.parametrize("bad_hour", [-1, 24, 100])
def test_invalid_hour_fails(bad_hour):
    from app.validators import validate_predict_input

    errors = validate_predict_input({
        "zone": "residential", "hour": bad_hour, "day_of_week": 3,
        "temperature": 20.0, "humidity": 55.0,
    })
    assert any("hour" in e for e in errors)


@pytest.mark.parametrize("bad_temp", [-25.0, 55.0])
def test_invalid_temperature_fails(bad_temp):
    from app.validators import validate_predict_input

    errors = validate_predict_input({
        "zone": "residential", "hour": 12, "day_of_week": 3,
        "temperature": bad_temp, "humidity": 55.0,
    })
    assert any("temperature" in e for e in errors)


def test_multiple_errors_collected():
    from app.validators import validate_predict_input

    errors = validate_predict_input({
        "zone": "underwater", "hour": 30, "day_of_week": 10,
        "temperature": 100.0, "humidity": 200.0,
    })
    assert len(errors) >= 4
