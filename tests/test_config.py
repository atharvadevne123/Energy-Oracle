"""Tests for application Settings configuration."""

from __future__ import annotations

import os


def test_settings_defaults():
    from app.config import Settings

    s = Settings()
    assert s.app_name == "Energy-Oracle"
    assert s.app_version == "1.1.0"
    assert isinstance(s.rate_limit_per_minute, int)
    assert s.rate_limit_per_minute > 0


def test_settings_blend_alpha_default():
    from app.config import Settings

    s = Settings()
    assert 0.0 < s.blend_alpha <= 1.0


def test_settings_cors_origins_default():
    from app.config import Settings

    s = Settings()
    assert isinstance(s.cors_origins, list)
    assert len(s.cors_origins) >= 1


def test_settings_cors_origins_from_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://example.com,https://app.example.com")
    from importlib import reload

    import app.config as cfg_module
    reload(cfg_module)
    s = cfg_module.Settings()
    assert "https://example.com" in s.cors_origins
    assert "https://app.example.com" in s.cors_origins
    reload(cfg_module)


def test_settings_rmse_threshold():
    from app.config import Settings

    s = Settings()
    assert s.rmse_threshold > 0


def test_settings_log_level_is_string():
    from app.config import Settings

    s = Settings()
    assert isinstance(s.log_level, str)
    assert s.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
