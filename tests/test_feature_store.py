"""Tests for the TTL-based in-memory feature store."""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def store():
    from app.feature_store import FeatureStore

    return FeatureStore(ttl=2)


def test_put_and_get(store):
    store.put("zone:residential:hour:10", {"temp": 20.0, "humidity": 55.0})
    result = store.get("zone:residential:hour:10")
    assert result == {"temp": 20.0, "humidity": 55.0}


def test_get_missing_returns_none(store):
    result = store.get("nonexistent_entity")
    assert result is None


def test_delete_existing(store):
    store.put("entity_1", {"val": 1})
    existed = store.delete("entity_1")
    assert existed
    assert store.get("entity_1") is None


def test_delete_missing_returns_false(store):
    assert not store.delete("never_existed")


def test_size_counts_active(store):
    store.put("a", {"x": 1})
    store.put("b", {"x": 2})
    assert store.size() == 2


def test_ttl_expiry(store):
    store.put("expiring", {"data": "test"})
    time.sleep(2.1)
    assert store.get("expiring") is None


def test_evict_expired(store):
    store.put("e1", {"v": 1})
    store.put("e2", {"v": 2})
    time.sleep(2.1)
    n_evicted = store.evict_expired()
    assert n_evicted == 2
    assert store.size() == 0
