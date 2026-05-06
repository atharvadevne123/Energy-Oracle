"""Tests for zone consumption profiles."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("zone", ["residential", "commercial", "industrial", "mixed"])
def test_all_zones_registered(zone):
    from app.zone_profiles import ALL_ZONES
    assert zone in ALL_ZONES


def test_peak_hour_higher_than_off_peak():
    from app.zone_profiles import get_zone_profile

    profile = get_zone_profile("residential")
    peak = profile.expected_kwh(hour=19, is_weekend=False, temperature=20.0)
    off_peak = profile.expected_kwh(hour=3, is_weekend=False, temperature=20.0)
    assert peak > off_peak


def test_weekend_adjustment_commercial():
    from app.zone_profiles import get_zone_profile

    profile = get_zone_profile("commercial")
    weekday = profile.expected_kwh(hour=12, is_weekend=False, temperature=20.0)
    weekend = profile.expected_kwh(hour=12, is_weekend=True, temperature=20.0)
    # Commercial drops on weekends
    assert weekend < weekday


def test_temperature_sensitivity():
    from app.zone_profiles import get_zone_profile

    profile = get_zone_profile("industrial")
    cold = profile.expected_kwh(hour=12, is_weekend=False, temperature=5.0)
    hot = profile.expected_kwh(hour=12, is_weekend=False, temperature=35.0)
    # Hot temperature should increase consumption
    assert hot > cold


def test_unknown_zone_falls_back_to_mixed():
    from app.zone_profiles import get_zone_profile, MIXED

    profile = get_zone_profile("unknown_zone")
    assert profile.name == MIXED.name


def test_expected_kwh_never_negative():
    from app.zone_profiles import ALL_ZONES

    for zone, profile in ALL_ZONES.items():
        for hour in [0, 12, 19]:
            kwh = profile.expected_kwh(hour=hour, is_weekend=False, temperature=-10.0)
            assert kwh >= 0, f"Negative kWh for {zone} at hour {hour}"
