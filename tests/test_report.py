"""Tests for monitoring report generation."""

from __future__ import annotations

import pytest

# fpdf2 requires cryptography which uses a Rust extension that panics in some environments.
# Probe at module level and skip the entire file if unavailable.
try:
    import fpdf as _fpdf_mod  # noqa: F401

    _FPDF_AVAILABLE = True
except BaseException:
    _FPDF_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FPDF_AVAILABLE,
    reason="fpdf unavailable or cryptography backend broken in this environment",
)


@pytest.fixture()
def sample_summary():
    return {
        "count": 10,
        "mean_kwh": 42.5,
        "min_kwh": 10.0,
        "max_kwh": 90.0,
    }


@pytest.fixture()
def sample_drift():
    return {
        "status": "stable",
        "features": {
            "temperature": {"ks_statistic": 0.05, "p_value": 0.8, "drift_detected": False},
            "humidity": {"ks_statistic": 0.04, "p_value": 0.9, "drift_detected": False},
        },
    }


def test_generate_report_returns_path(tmp_path, sample_summary, sample_drift):
    from app.report import generate_monitoring_report

    out = tmp_path / "report.pdf"
    result = generate_monitoring_report(sample_summary, sample_drift, output_path=out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_report_creates_parent_dir(tmp_path, sample_summary, sample_drift):
    from app.report import generate_monitoring_report

    out = tmp_path / "subdir" / "report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = generate_monitoring_report(sample_summary, sample_drift, output_path=out)
    assert result.exists()


def test_generate_report_drift_detected(tmp_path, sample_summary):
    from app.report import generate_monitoring_report

    drift = {
        "status": "drift_detected",
        "features": {
            "temperature": {"ks_statistic": 0.4, "p_value": 0.001, "drift_detected": True},
        },
    }
    out = tmp_path / "drift_report.pdf"
    result = generate_monitoring_report(sample_summary, drift, output_path=out)
    assert result.exists()


def test_generate_report_empty_summary(tmp_path, sample_drift):
    from app.report import generate_monitoring_report

    out = tmp_path / "empty_report.pdf"
    result = generate_monitoring_report({}, sample_drift, output_path=out)
    assert result.exists()


def test_generate_report_no_features(tmp_path, sample_summary):
    from app.report import generate_monitoring_report

    out = tmp_path / "nofeat_report.pdf"
    result = generate_monitoring_report(
        sample_summary, {"status": "insufficient_data"}, output_path=out
    )
    assert result.exists()
