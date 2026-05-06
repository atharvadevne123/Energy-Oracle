"""Simple in-process LRU cache for model predictions."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_prediction_cache: dict[str, float] = {}
_CACHE_MAX_SIZE = 512


def _make_cache_key(data: dict[str, Any]) -> str:
    """Create a deterministic cache key from prediction input."""
    serialised = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def cached_predict(data: dict[str, Any], predict_fn: Any) -> tuple[float, bool]:
    """
    Return a cached prediction if available, otherwise call predict_fn.

    Returns:
        Tuple of (predicted_kwh, cache_hit).
    """
    key = _make_cache_key(data)
    if key in _prediction_cache:
        logger.debug("Cache hit key=%s", key)
        return _prediction_cache[key], True

    result = predict_fn(data)
    if len(_prediction_cache) >= _CACHE_MAX_SIZE:
        # Evict oldest 10 % on overflow
        evict = list(_prediction_cache.keys())[: _CACHE_MAX_SIZE // 10]
        for k in evict:
            del _prediction_cache[k]
    _prediction_cache[key] = result
    logger.debug("Cache miss key=%s — stored result %.4f", key, result)
    return result, False


def clear_cache() -> int:
    """Clear the prediction cache and return number of evicted entries."""
    n = len(_prediction_cache)
    _prediction_cache.clear()
    return n


def cache_stats() -> dict[str, int]:
    """Return current cache utilisation."""
    return {"size": len(_prediction_cache), "max_size": _CACHE_MAX_SIZE}
