"""Simple in-process LRU cache for model predictions."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["cached_predict", "clear_cache", "cache_stats", "CACHE_MAX_SIZE"]

CACHE_MAX_SIZE = 512

_prediction_cache: OrderedDict[str, float] = OrderedDict()


def _make_cache_key(data: dict[str, Any]) -> str:
    """Create a deterministic cache key from prediction input."""
    serialised = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def cached_predict(data: dict[str, Any], predict_fn: Any) -> tuple[float, bool]:
    """
    Return a cached prediction if available, otherwise call predict_fn.

    Uses an OrderedDict to implement true LRU eviction — the least recently
    used entry is removed when the cache exceeds CACHE_MAX_SIZE.

    Returns:
        Tuple of (predicted_kwh, cache_hit).
    """
    key = _make_cache_key(data)
    if key in _prediction_cache:
        _prediction_cache.move_to_end(key)
        logger.debug("Cache hit key=%s", key)
        return _prediction_cache[key], True

    result = predict_fn(data)
    _prediction_cache[key] = result
    _prediction_cache.move_to_end(key)
    if len(_prediction_cache) > CACHE_MAX_SIZE:
        evicted_key, _ = _prediction_cache.popitem(last=False)
        logger.debug("LRU eviction key=%s", evicted_key)
    logger.debug("Cache miss key=%s — stored result %.4f", key, result)
    return result, False


def clear_cache() -> int:
    """Clear the prediction cache and return number of evicted entries."""
    n = len(_prediction_cache)
    _prediction_cache.clear()
    return n


def cache_stats() -> dict[str, int]:
    """Return current cache utilisation."""
    return {"size": len(_prediction_cache), "max_size": CACHE_MAX_SIZE}
