"""Tests for in-process metrics store."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_metrics():
    from app.metrics_store import reset

    reset()
    yield
    reset()


def test_increment_counter():
    from app.metrics_store import get_counter, increment

    increment("predictions_total")
    increment("predictions_total")
    assert get_counter("predictions_total") == 2


def test_counter_with_labels():
    from app.metrics_store import get_counter, increment

    increment("requests", labels={"zone": "residential"})
    increment("requests", labels={"zone": "commercial"})
    # Keys with labels are distinct from the base key
    assert get_counter("requests") == 0


def test_record_and_summarise_latency():
    from app.metrics_store import record_latency, summary

    for ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
        record_latency("/predict", ms)

    s = summary()
    assert "/predict" in s["latency"]
    stats = s["latency"]["/predict"]
    assert stats["count"] == 5
    assert stats["mean_ms"] == 30.0


def test_percentile_calculation():
    from app.metrics_store import percentile

    values = list(range(1, 101))  # 1..100
    assert percentile(values, 95) == 95
    assert percentile(values, 50) == 50


def test_summary_includes_uptime():
    from app.metrics_store import summary

    s = summary()
    assert "uptime_seconds" in s
    assert s["uptime_seconds"] >= 0


def test_reset_clears_all():
    from app.metrics_store import increment, reset, summary

    increment("some_counter")
    reset()
    s = summary()
    assert s["counters"] == {}
    assert s["latency"] == {}
