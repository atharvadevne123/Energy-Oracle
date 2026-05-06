"""Tests for health check utilities."""

from __future__ import annotations


def test_check_database_ok():
    from app.health import check_database

    result = check_database("sqlite:///:memory:")
    assert result["status"] == "ok"


def test_check_database_bad_url():
    from app.health import check_database

    result = check_database("postgresql://fake:fake@localhost:9999/nodb")
    assert result["status"] == "error"
    assert "detail" in result


def test_check_model_missing(tmp_path):
    from app.health import check_model

    result = check_model(tmp_path / "nonexistent.joblib")
    assert result["status"] == "missing"


def test_check_model_ok(tmp_path):
    from app.health import check_model

    model_file = tmp_path / "test.joblib"
    model_file.write_bytes(b"fake model data" * 100)
    result = check_model(model_file)
    assert result["status"] == "ok"
    assert "size_kb" in result


def test_full_health_report_degraded_without_model(tmp_path):
    from app.health import full_health_report

    result = full_health_report(
        db_url="sqlite:///:memory:",
        model_path=tmp_path / "missing.joblib",
    )
    assert result["status"] == "degraded"
    assert "components" in result
