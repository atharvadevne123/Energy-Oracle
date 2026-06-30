"""Lightweight in-memory feature store for online serving."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ['FeatureStore', 'put', 'get', 'delete', 'evict_expired', 'size']



# TTL in seconds for feature store entries
DEFAULT_TTL = 3600


class FeatureStore:
    """
    Simple TTL-based in-memory feature store.

    In production, replace with Redis or a managed feature platform.
    """

    def __init__(self, ttl: int = DEFAULT_TTL) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._ttl = ttl

    def put(self, entity_id: str, features: dict[str, Any]) -> None:
        """
        Store features for an entity.

        Args:
            entity_id: Unique identifier (e.g. zone:hour hash).
            features: Feature dict to cache.
        """
        self._store[entity_id] = {
            "features": features,
            "expires_at": time.time() + self._ttl,
        }
        logger.debug("Feature store PUT entity=%s", entity_id)

    def get(self, entity_id: str) -> dict[str, Any] | None:
        """
        Retrieve features for an entity if not expired.

        Args:
            entity_id: Entity to look up.

        Returns:
            Feature dict or None if missing/expired.
        """
        entry = self._store.get(entity_id)
        if entry is None:
            return None
        if time.time() > entry["expires_at"]:
            del self._store[entity_id]
            logger.debug("Feature store TTL expired entity=%s", entity_id)
            return None
        return entry["features"]

    def delete(self, entity_id: str) -> bool:
        """Remove an entity from the store. Returns True if it existed."""
        existed = entity_id in self._store
        self._store.pop(entity_id, None)
        return existed

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count evicted."""
        now = time.time()
        expired = [k for k, v in self._store.items() if now > v["expires_at"]]
        for k in expired:
            del self._store[k]
        return len(expired)

    def size(self) -> int:
        """Return number of active (non-expired) entries."""
        self.evict_expired()
        return len(self._store)


# Module-level default store instance
feature_store = FeatureStore()
