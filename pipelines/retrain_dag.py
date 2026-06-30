"""Automated retraining pipeline — runnable standalone or as an Airflow DAG."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core retraining logic (no Airflow dependency)
# ---------------------------------------------------------------------------


def retrain(n_samples: int = 5000, cv_folds: int = 5) -> dict:
    """
    Pull latest data (synthetic fallback), retrain ensemble, persist metrics.

    In production, replace generate_synthetic_data() with a database query
    or S3 fetch to get fresh labelled examples.
    """
    from app.model import generate_synthetic_data, train_model

    run_id = str(uuid.uuid4())
    logger.info("Retraining job started — run_id=%s", run_id)

    df, y = generate_synthetic_data(n_samples=n_samples)
    pipe, metrics = train_model(df, y, cv_folds=cv_folds)

    metrics["run_id"] = run_id
    metrics["retrained_at"] = datetime.now(tz=UTC).isoformat()

    log_path = Path("retrain_history.json")
    history: list[dict] = []
    if log_path.exists():
        history = json.loads(log_path.read_text())
    history.append(metrics)
    # Keep last 50 runs
    log_path.write_text(json.dumps(history[-50:], indent=2))

    logger.info(
        "Retraining complete — RMSE=%.4f ± %.4f",
        metrics["rmse_mean"],
        metrics["rmse_std"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Airflow DAG (only parsed when Airflow is installed)
# ---------------------------------------------------------------------------

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _default_args = {
        "owner": "reflective-lantern",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="energy_oracle_retrain",
        default_args=_default_args,
        description="Weekly LightGBM retraining for Energy-Oracle",
        schedule_interval="@weekly",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["ml", "energy", "retrain"],
    ) as dag:

        def _retrain_task(**context):
            metrics = retrain(n_samples=10000, cv_folds=5)
            context["task_instance"].xcom_push(key="metrics", value=metrics)

        def _validate_task(**context):
            ti = context["task_instance"]
            metrics = ti.xcom_pull(key="metrics", task_ids="retrain_model")
            threshold = float(os.getenv("RMSE_THRESHOLD", "15.0"))
            if metrics["rmse_mean"] > threshold:
                raise ValueError(
                    f"RMSE {metrics['rmse_mean']:.4f} exceeds threshold {threshold}. "
                    "Retraining rejected."
                )
            logger.info("Model validation passed — RMSE=%.4f", metrics["rmse_mean"])

        retrain_task = PythonOperator(
            task_id="retrain_model",
            python_callable=_retrain_task,
        )

        validate_task = PythonOperator(
            task_id="validate_model",
            python_callable=_validate_task,
        )

        retrain_task >> validate_task

except ImportError:
    # Airflow not installed — DAG definition is skipped silently
    dag = None  # type: ignore[assignment]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = retrain()
    print(json.dumps(result, indent=2))
