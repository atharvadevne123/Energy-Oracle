"""Tests for prediction caching module."""

from __future__ import annotations


def test_cache_hit_on_repeat_call():
    from app.cache import cached_predict, clear_cache

    clear_cache()
    call_count = 0

    def mock_predict(data):
        nonlocal call_count
        call_count += 1
        return 42.0

    data = {
        "zone": "residential",
        "hour": 10,
        "day_of_week": 2,
        "temperature": 20.0,
        "humidity": 50.0,
    }
    result1, hit1 = cached_predict(data, mock_predict)
    result2, hit2 = cached_predict(data, mock_predict)

    assert result1 == result2 == 42.0
    assert not hit1
    assert hit2
    assert call_count == 1  # predict_fn only called once


def test_different_inputs_different_cache_entries():
    from app.cache import cached_predict, clear_cache

    clear_cache()
    results = []
    counter = [0]

    def mock_predict(data):
        counter[0] += 1
        return float(counter[0] * 10)

    for hour in [8, 12, 18]:
        data = {
            "zone": "commercial",
            "hour": hour,
            "day_of_week": 1,
            "temperature": 22.0,
            "humidity": 60.0,
        }
        result, _ = cached_predict(data, mock_predict)
        results.append(result)

    assert len(set(results)) == 3
    assert counter[0] == 3


def test_cache_stats():
    from app.cache import cache_stats, clear_cache

    clear_cache()
    stats = cache_stats()
    assert stats["size"] == 0
    assert stats["max_size"] > 0


def test_clear_cache_returns_count():
    from app.cache import cached_predict, clear_cache

    clear_cache()
    data = {"zone": "mixed", "hour": 6, "day_of_week": 0, "temperature": 15.0, "humidity": 70.0}
    cached_predict(data, lambda d: 55.0)
    n = clear_cache()
    assert n >= 1
