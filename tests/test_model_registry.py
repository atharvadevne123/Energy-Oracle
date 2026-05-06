"""Tests for file-based model registry."""

from __future__ import annotations

import pytest


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    import app.model_registry as reg_mod
    monkeypatch.setattr(reg_mod, "REGISTRY_PATH", tmp_path / "registry.json")
    return tmp_path / "registry.json"


def test_register_creates_entry(registry_path):
    from app.model_registry import load_registry, register_model

    register_model("1.0.0", {"rmse_mean": 5.0}, "model.joblib")
    registry = load_registry()
    assert len(registry) == 1
    assert registry[0]["version"] == "1.0.0"


def test_load_empty_registry(registry_path):
    from app.model_registry import load_registry
    assert load_registry() == []


def test_promote_sets_production(registry_path):
    from app.model_registry import get_production_model, promote, register_model

    register_model("1.0.0", {}, "model.joblib")
    register_model("2.0.0", {}, "model_v2.joblib")
    result = promote("2.0.0")
    assert result
    prod = get_production_model()
    assert prod["version"] == "2.0.0"


def test_promote_demotes_others(registry_path):
    from app.model_registry import load_registry, promote, register_model

    register_model("1.0.0", {}, "m1.joblib", promoted=True)
    register_model("2.0.0", {}, "m2.joblib")
    promote("2.0.0")

    registry = load_registry()
    versions = {r["version"]: r["promoted"] for r in registry}
    assert versions["1.0.0"] is False
    assert versions["2.0.0"] is True


def test_promote_nonexistent_returns_false(registry_path):
    from app.model_registry import promote
    assert not promote("99.0.0")


def test_get_production_empty(registry_path):
    from app.model_registry import get_production_model
    assert get_production_model() is None
