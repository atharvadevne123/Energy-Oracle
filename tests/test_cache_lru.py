"""Tests for LRU prediction cache behaviour."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_cache():
    from app.cache import clear_cache

    clear_cache()
    yield
    clear_cache()


def test_cache_miss_calls_predict_fn():
    from app.cache import cached_predict

    calls = []

    def predict_fn(data):
        calls.append(data)
        return 42.0

    data = {
        "zone": "residential",
        "hour": 8,
        "day_of_week": 1,
        "temperature": 20.0,
        "humidity": 50.0,
    }
    result, hit = cached_predict(data, predict_fn)
    assert result == 42.0
    assert hit is False
    assert len(calls) == 1


def test_cache_hit_skips_predict_fn():
    from app.cache import cached_predict

    calls = []

    def predict_fn(data):
        calls.append(data)
        return 99.0

    data = {
        "zone": "commercial",
        "hour": 12,
        "day_of_week": 2,
        "temperature": 22.0,
        "humidity": 55.0,
    }
    cached_predict(data, predict_fn)
    result, hit = cached_predict(data, predict_fn)
    assert result == 99.0
    assert hit is True
    assert len(calls) == 1


def test_cache_lru_eviction():
    from app.cache import CACHE_MAX_SIZE, cache_stats, cached_predict

    def predict_fn(data):
        return float(data["hour"])

    for i in range(CACHE_MAX_SIZE + 5):
        cached_predict(
            {"hour": i, "zone": "mixed", "day_of_week": 0, "temperature": 20.0, "humidity": 50.0},
            predict_fn,
        )

    stats = cache_stats()
    assert stats["size"] <= CACHE_MAX_SIZE


def test_cache_stats_max_size():
    from app.cache import CACHE_MAX_SIZE, cache_stats

    stats = cache_stats()
    assert stats["max_size"] == CACHE_MAX_SIZE


def test_clear_cache_returns_count():
    from app.cache import cached_predict, clear_cache

    for i in range(3):
        cached_predict(
            {"hour": i, "zone": "mixed", "day_of_week": 0, "temperature": 20.0, "humidity": 50.0},
            lambda d: 1.0,
        )

    n = clear_cache()
    assert n == 3
