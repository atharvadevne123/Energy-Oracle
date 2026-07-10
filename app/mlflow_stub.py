"""
MLflow-compatible experiment tracking stub.

When MLflow is installed, delegates to it. Otherwise logs to a local JSON file.
This pattern allows seamless promotion from dev → production MLflow servers.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["start_run"]


_LOG_PATH = Path("mlflow_runs.json")


class _LocalRun:
    """Fallback run context manager when MLflow is not installed."""

    def __init__(self, experiment: str, run_name: str) -> None:
        self.run_id = str(uuid.uuid4())
        self.experiment = experiment
        self.run_name = run_name
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._tags: dict[str, str] = {}
        self._start = time.time()

    def log_param(self, key: str, value: Any) -> None:
        self._params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        self._metrics[key] = value

    def set_tag(self, key: str, value: str) -> None:
        self._tags[key] = value

    def __enter__(self) -> _LocalRun:
        logger.info("MLflow stub: starting run %s", self.run_id)
        return self

    def __exit__(self, *args: Any) -> None:
        duration = round(time.time() - self._start, 2)
        record = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "run_name": self.run_name,
            "params": self._params,
            "metrics": self._metrics,
            "tags": self._tags,
            "duration_seconds": duration,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        history: list[dict] = []
        if _LOG_PATH.exists():
            history = json.loads(_LOG_PATH.read_text())
        history.append(record)
        _LOG_PATH.write_text(json.dumps(history[-100:], indent=2))
        logger.info("MLflow stub: run %s logged to %s", self.run_id, _LOG_PATH)


def start_run(experiment: str = "energy-oracle", run_name: str = "train") -> _LocalRun:
    """Return a run context manager (MLflow or local fallback)."""
    try:
        import mlflow  # type: ignore[import]

        mlflow.set_experiment(experiment)
        return mlflow.start_run(run_name=run_name)  # type: ignore[return-value]
    except ImportError:
        return _LocalRun(experiment=experiment, run_name=run_name)
