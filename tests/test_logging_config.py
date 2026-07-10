"""Tests for structured JSON logging configuration."""

from __future__ import annotations

import json
import logging


def test_configure_logging_sets_level():
    from app.logging_config import configure_logging

    configure_logging(level="WARNING")
    assert logging.getLogger().level == logging.WARNING
    configure_logging(level="INFO")


def test_json_formatter_produces_valid_json():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    payload = json.loads(output)
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert "ts" in payload


def test_json_formatter_includes_exception():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="error occurred",
        args=(),
        exc_info=exc_info,
    )
    output = formatter.format(record)
    payload = json.loads(output)
    assert "exception" in payload
    assert "ValueError" in payload["exception"]


def test_configure_logging_plain_text_formatter():
    from app.logging_config import configure_logging

    configure_logging(level="DEBUG", json_logs=False)
    root = logging.getLogger()
    handler = root.handlers[0]
    assert not isinstance(handler.formatter, type(None))
    configure_logging(level="INFO")
