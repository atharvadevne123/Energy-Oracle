"""Tests for the retraining pipeline."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def patched_model(tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_retrain_returns_metrics(patched_model):
    from pipelines.retrain_dag import retrain

    metrics = retrain(n_samples=300, cv_folds=2)
    assert "rmse_mean" in metrics
    assert "rmse_std" in metrics
    assert "run_id" in metrics
    assert "retrained_at" in metrics


def test_retrain_rmse_positive(patched_model):
    from pipelines.retrain_dag import retrain

    metrics = retrain(n_samples=300, cv_folds=2)
    assert metrics["rmse_mean"] > 0


def test_retrain_writes_history_file(patched_model):
    from pipelines.retrain_dag import retrain

    retrain(n_samples=300, cv_folds=2)
    history_file = patched_model / "retrain_history.json"
    assert history_file.exists()
    history = json.loads(history_file.read_text())
    assert isinstance(history, list)
    assert len(history) == 1


def test_retrain_accumulates_history(patched_model):
    from pipelines.retrain_dag import retrain

    retrain(n_samples=200, cv_folds=2)
    retrain(n_samples=200, cv_folds=2)
    history_file = patched_model / "retrain_history.json"
    history = json.loads(history_file.read_text())
    assert len(history) == 2


def test_retrain_history_has_unique_run_ids(patched_model):
    from pipelines.retrain_dag import retrain

    m1 = retrain(n_samples=200, cv_folds=2)
    m2 = retrain(n_samples=200, cv_folds=2)
    assert m1["run_id"] != m2["run_id"]


def test_retrain_retrained_at_is_iso_string(patched_model):
    from datetime import datetime

    from pipelines.retrain_dag import retrain

    metrics = retrain(n_samples=200, cv_folds=2)
    ts = metrics["retrained_at"]
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
