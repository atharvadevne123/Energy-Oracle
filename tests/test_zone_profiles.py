"""Tests for zone consumption profiles."""

from __future__ import annotations

import pytest


def test_all_zones_available():
    from app.zone_profiles import ALL_ZONES

    assert set(ALL_ZONES) == {"residential", "commercial", "industrial", "mixed"}


def test_get_zone_profile_residential():
    from app.zone_profiles import get_zone_profile

    profile = get_zone_profile("residential")
    assert profile.name == "residential"
    assert profile.baseline_kwh > 0


def test_get_zone_profile_commercial():
    from app.zone_profiles import get_zone_profile

    profile = get_zone_profile("commercial")
    assert profile.baseline_kwh > get_zone_profile("residential").baseline_kwh


def test_get_zone_profile_industrial_largest():
    from app.zone_profiles import get_zone_profile

    industrial = get_zone_profile("industrial")
    residential = get_zone_profile("residential")
    commercial = get_zone_profile("commercial")
    assert industrial.baseline_kwh > commercial.baseline_kwh > residential.baseline_kwh


def test_get_zone_profile_unknown_raises():
    from app.zone_profiles import get_zone_profile

    with pytest.raises(KeyError):
        get_zone_profile("underwater")


def test_zone_profile_peak_multiplier_gt_one():
    from app.zone_profiles import RESIDENTIAL

    assert RESIDENTIAL.peak_multiplier > 1.0


def test_expected_kwh_peak_higher_than_off_peak():
    from app.zone_profiles import INDUSTRIAL

    peak = INDUSTRIAL.expected_kwh(hour=19, is_weekend=False, temperature=20.0)
    off_peak = INDUSTRIAL.expected_kwh(hour=3, is_weekend=False, temperature=20.0)
    assert peak > off_peak


def test_expected_kwh_non_negative():
    from app.zone_profiles import ALL_ZONES, get_zone_profile

    for zone_name in ALL_ZONES:
        p = get_zone_profile(zone_name)
        val = p.expected_kwh(hour=2, is_weekend=True, temperature=-10.0)
        assert val >= 0.0, f"{zone_name} expected_kwh negative"
