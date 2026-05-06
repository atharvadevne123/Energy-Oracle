"""Centralised application configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Application settings loaded from environment variables."""

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./energy_oracle.db")
    model_path: Path = Path(os.getenv("MODEL_PATH", "model.joblib"))
    metrics_path: Path = Path(os.getenv("METRICS_PATH", "metrics.json"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    rmse_threshold: float = float(os.getenv("RMSE_THRESHOLD", "20.0"))
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
    app_name: str = "Energy-Oracle"
    app_version: str = "1.0.0"
    blend_alpha: float = float(os.getenv("BLEND_ALPHA", "0.7"))


settings = Settings()
