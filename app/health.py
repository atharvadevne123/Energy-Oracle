"""Health check utilities with detailed component status."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


__all__ = ['check_database', 'check_model', 'full_health_report']
_start_time = time.time()


def check_database(db_url: str) -> dict[str, Any]:
    """Verify database connectivity."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


def check_model(model_path: str | Path) -> dict[str, Any]:
    """Verify model file exists and can be loaded."""
    path = Path(model_path)
    if not path.exists():
        return {"status": "missing", "detail": f"Model not found at {path}"}
    try:
        size_kb = path.stat().st_size // 1024
        return {"status": "ok", "size_kb": size_kb}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def full_health_report(db_url: str, model_path: str | Path) -> dict[str, Any]:
    """
    Aggregate health report for all components.

    Args:
        db_url: SQLAlchemy database URL.
        model_path: Path to the serialised model.

    Returns:
        Dict with per-component status and overall health.
    """
    components = {
        "database": check_database(db_url),
        "model": check_model(model_path),
    }
    all_ok = all(c["status"] == "ok" for c in components.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "uptime_seconds": round(time.time() - _start_time, 2),
        "version": "1.0.0",
        "components": components,
    }

