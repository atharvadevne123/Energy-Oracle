"""Tests for validate_batch_input helper."""

from __future__ import annotations


def test_validate_batch_input_all_valid():
    from app.validators import validate_batch_input

    records = [
        {"zone": "residential", "hour": 8, "day_of_week": 1, "temperature": 20.0, "humidity": 55.0},
        {"zone": "commercial", "hour": 12, "day_of_week": 3, "temperature": 22.0, "humidity": 60.0},
    ]
    errors = validate_batch_input(records)
    assert errors == {}


def test_validate_batch_input_one_invalid():
    from app.validators import validate_batch_input

    records = [
        {"zone": "residential", "hour": 8, "day_of_week": 1, "temperature": 20.0, "humidity": 55.0},
        {"zone": "bad_zone", "hour": 8, "day_of_week": 1, "temperature": 20.0, "humidity": 55.0},
    ]
    errors = validate_batch_input(records)
    assert 1 in errors
    assert 0 not in errors


def test_validate_batch_input_missing_field():
    from app.validators import validate_batch_input

    records = [{"zone": "residential", "hour": 8}]
    errors = validate_batch_input(records)
    assert 0 in errors
    assert len(errors[0]) >= 1


def test_validate_batch_input_empty():
    from app.validators import validate_batch_input

    errors = validate_batch_input([])
    assert errors == {}


def test_valid_zones_set():
    from app.validators import VALID_ZONES

    assert "residential" in VALID_ZONES
    assert "commercial" in VALID_ZONES
    assert "industrial" in VALID_ZONES
    assert "mixed" in VALID_ZONES
    assert "underwater" not in VALID_ZONES
