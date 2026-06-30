"""Extended tests for seasonality module."""

from __future__ import annotations

import math

import pytest


@pytest.mark.parametrize("month,expected_season", [
    (1, "winter"), (2, "winter"), (12, "winter"),
    (3, "spring"), (4, "spring"), (5, "spring"),
    (6, "summer"), (7, "summer"), (8, "summer"),
    (9, "autumn"), (10, "autumn"), (11, "autumn"),
])
def test_month_to_season(month, expected_season):
    from app.seasonality import month_to_season
    assert month_to_season(month) == expected_season


def test_season_encoding_returns_int():
    from app.seasonality import season_encoding
    for season in ("winter", "spring", "summer", "autumn"):
        enc = season_encoding(season)
        assert isinstance(enc, int)
        assert 0 <= enc <= 3


def test_cyclical_month_returns_two_floats():
    from app.seasonality import cyclical_month
    sin_v, cos_v = cyclical_month(6)
    assert -1.0 <= sin_v <= 1.0
    assert -1.0 <= cos_v <= 1.0


def test_cyclical_month_unit_circle():
    from app.seasonality import cyclical_month
    for month in range(1, 13):
        sin_v, cos_v = cyclical_month(month)
        assert math.isclose(sin_v ** 2 + cos_v ** 2, 1.0, rel_tol=1e-6)


def test_get_seasonal_features_keys():
    from app.seasonality import get_seasonal_features
    feats = get_seasonal_features(month=7, day=4)
    assert "season_encoded" in feats
    assert "month_sin" in feats
    assert "month_cos" in feats
