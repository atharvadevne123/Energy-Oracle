"""Tests for health check utilities."""

from __future__ import annotations

import pytest


def test_check_database_ok(tmp_path):
    from app.health import check_database

    db_url = f"sqlite:///{tmp_path}/test.db"
    result = check_database(db_url)
    assert result["status"] == "ok"


def test_check_database_bad_url():
    from app.health import check_database

    result = check_database("invalid://not_a_real_db")
    assert result["status"] in ("error", "missing")


def test_check_model_ok(tmp_path):
    import joblib
    from app.health import check_model

    model_path = tmp_path / "model.joblib"
    joblib.dump({"test": True}, model_path)
    result = check_model(model_path)
    assert result["status"] == "ok"
    assert "size_kb" in result


def test_check_model_missing(tmp_path):
    from app.health import check_model

    result = check_model(tmp_path / "nonexistent.joblib")
    assert result["status"] == "missing"


def test_full_health_report_structure(tmp_path):
    import joblib
    from app.health import full_health_report

    model_path = tmp_path / "model.joblib"
    joblib.dump({"test": True}, model_path)
    db_url = f"sqlite:///{tmp_path}/health_test.db"

    report = full_health_report(db_url=db_url, model_path=model_path)
    assert "status" in report
    assert report["status"] in ("ok", "degraded")
    assert "components" in report
    assert "database" in report["components"]
    assert "model" in report["components"]
    assert "uptime_seconds" in report
