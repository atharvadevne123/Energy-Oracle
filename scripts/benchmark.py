"""Benchmark API latency against a running Energy-Oracle instance."""

from __future__ import annotations

import statistics
import sys
import time
import urllib.request
import json
import argparse

DEFAULT_URL = "http://localhost:8000/api/v1/predict"
DEFAULT_N = 50

PAYLOAD = json.dumps({
    "zone": "commercial",
    "hour": 14,
    "day_of_week": 3,
    "temperature": 22.5,
    "humidity": 58.0,
}).encode()


def run_benchmark(url: str, n: int) -> dict:
    """
    Send n POST requests and collect latency statistics.

    Args:
        url: Full endpoint URL.
        n: Number of requests.

    Returns:
        Dict with min, max, mean, p95, p99 latencies in milliseconds.
    """
    latencies: list[float] = []
    errors = 0

    for i in range(n):
        req = urllib.request.Request(
            url, data=PAYLOAD, method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            errors += 1
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

    if not latencies:
        return {"error": "All requests failed"}

    latencies_sorted = sorted(latencies)
    p95_idx = int(0.95 * len(latencies_sorted))
    p99_idx = int(0.99 * len(latencies_sorted))

    return {
        "n_requests": n,
        "n_errors": errors,
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p95_ms": round(latencies_sorted[p95_idx], 2),
        "p99_ms": round(latencies_sorted[p99_idx], 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Energy-Oracle predict endpoint")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()

    print(f"Benchmarking {args.url} with {args.n} requests...")
    results = run_benchmark(args.url, args.n)
    print(json.dumps(results, indent=2))
