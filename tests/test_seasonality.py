"""Tests for seasonality and calendar feature utilities."""

from __future__ import annotations

import math

import pytest


@pytest.mark.parametrize(
    "month,expected_season",
    [
        (1, "winter"),
        (2, "winter"),
        (12, "winter"),
        (3, "spring"),
        (5, "spring"),
        (6, "summer"),
        (8, "summer"),
        (9, "autumn"),
        (11, "autumn"),
    ],
)
def test_month_to_season(month, expected_season):
    from app.seasonality import month_to_season

    assert month_to_season(month) == expected_season


def test_cyclical_month_december_january_closeness():
    from app.seasonality import cyclical_month

    sin_dec, cos_dec = cyclical_month(12)
    sin_jan, cos_jan = cyclical_month(1)
    # Euclidean distance should be small (wrapping adjacency)
    dist = math.sqrt((sin_dec - sin_jan) ** 2 + (cos_dec - cos_jan) ** 2)
    assert dist < 0.6


def test_cyclical_encoding_range():
    from app.seasonality import cyclical_month

    for month in range(1, 13):
        sin, cos = cyclical_month(month)
        assert -1.01 <= sin <= 1.01
        assert -1.01 <= cos <= 1.01


def test_is_christmas_holiday():
    from app.seasonality import get_seasonal_features

    features = get_seasonal_features(month=12, day=25, year=2025)
    assert features["is_holiday"] == 1.0


def test_regular_day_not_holiday():
    from app.seasonality import get_seasonal_features

    features = get_seasonal_features(month=6, day=15, year=2025)
    assert features["is_holiday"] == 0.0


def test_get_seasonal_features_keys():
    from app.seasonality import get_seasonal_features

    features = get_seasonal_features(month=7, day=4, year=2025)
    for key in ["season_encoded", "month_sin", "month_cos", "is_holiday"]:
        assert key in features
