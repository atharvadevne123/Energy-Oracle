"""
AWS/boto3 integration stubs for model artifact upload and S3 data ingestion.

In production, set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION.
When boto3 is unavailable, all operations log a warning and return gracefully.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["upload_model", "download_training_data"]


S3_BUCKET = os.getenv("S3_BUCKET", "energy-oracle-artifacts")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def upload_model(model_path: str | Path, s3_key: str | None = None) -> dict[str, Any]:
    """
    Upload a model artifact to S3.

    Args:
        model_path: Local path to model.joblib.
        s3_key: Target S3 object key. Defaults to 'models/<filename>'.

    Returns:
        Dict with 'status' and optionally 's3_uri'.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return {"status": "error", "detail": f"Model not found: {model_path}"}

    key = s3_key or f"models/{model_path.name}"

    try:
        import boto3  # type: ignore[import]

        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.upload_file(str(model_path), S3_BUCKET, key)
        uri = f"s3://{S3_BUCKET}/{key}"
        logger.info("Model uploaded to %s", uri)
        return {"status": "ok", "s3_uri": uri}
    except ImportError:
        logger.warning("boto3 not installed — skipping S3 upload")
        return {"status": "skipped", "detail": "boto3 not installed"}
    except Exception as exc:
        logger.error("S3 upload failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


def download_training_data(s3_key: str, local_path: str | Path) -> dict[str, Any]:
    """
    Download training data from S3.

    Args:
        s3_key: S3 object key for the training dataset.
        local_path: Local destination path.

    Returns:
        Dict with 'status' and 'local_path' on success.
    """
    local_path = Path(local_path)
    try:
        import boto3  # type: ignore[import]

        s3 = boto3.client("s3", region_name=AWS_REGION)
        s3.download_file(S3_BUCKET, s3_key, str(local_path))
        logger.info("Training data downloaded to %s", local_path)
        return {"status": "ok", "local_path": str(local_path)}
    except ImportError:
        logger.warning("boto3 not installed — skipping S3 download")
        return {"status": "skipped", "detail": "boto3 not installed"}
    except Exception as exc:
        logger.error("S3 download failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
