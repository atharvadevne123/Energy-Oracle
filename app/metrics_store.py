"""Prometheus-compatible metrics counters for operational observability."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ['increment', 'record_latency', 'get_counter', 'percentile', 'summary', 'reset']



# In-process counters (replace with prometheus_client in production)
_counters: dict[str, int] = defaultdict(int)
_histograms: dict[str, list[float]] = defaultdict(list)
_start_time = time.time()


def increment(name: str, labels: dict[str, str] | None = None) -> None:
    """Increment a named counter."""
    key = name if not labels else f"{name}{{{','.join(f'{k}={v}' for k, v in sorted(labels.items()))}}}"
    _counters[key] += 1


def record_latency(endpoint: str, latency_ms: float) -> None:
    """Record a request latency sample for an endpoint."""
    _histograms[endpoint].append(latency_ms)
    if len(_histograms[endpoint]) > 10000:
        # Keep last 10k samples
        _histograms[endpoint] = _histograms[endpoint][-10000:]


def get_counter(name: str) -> int:
    """Return the current value of a counter."""
    return _counters.get(name, 0)


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(p / 100 * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def summary() -> dict[str, Any]:
    """Return a snapshot of all metrics."""
    latency_stats: dict[str, Any] = {}
    for endpoint, samples in _histograms.items():
        if samples:
            latency_stats[endpoint] = {
                "count": len(samples),
                "mean_ms": round(sum(samples) / len(samples), 2),
                "p95_ms": round(percentile(samples, 95), 2),
                "p99_ms": round(percentile(samples, 99), 2),
            }

    return {
        "uptime_seconds": round(time.time() - _start_time, 2),
        "counters": dict(_counters),
        "latency": latency_stats,
    }


def reset() -> None:
    """Clear all metrics (for testing)."""
    _counters.clear()
    _histograms.clear()
