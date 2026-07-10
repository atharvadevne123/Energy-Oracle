"""Structured JSON logging configuration for production deployments."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

__all__ = ["JsonFormatter", "configure_logging"]


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines for log-aggregation pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the log record to a compact JSON string."""
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "correlation_id"):
            payload["correlation_id"] = record.correlation_id
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """
    Configure root logger.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        json_logs: When True, emit structured JSON lines instead of plain text.
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
