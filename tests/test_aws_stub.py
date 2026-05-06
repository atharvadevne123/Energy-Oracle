"""Tests for AWS/S3 stub (without live AWS credentials)."""

from __future__ import annotations

import pytest


def test_upload_missing_model_returns_error(tmp_path):
    from app.aws_stub import upload_model
    result = upload_model(tmp_path / "nonexistent.joblib")
    assert result["status"] == "error"
    assert "not found" in result["detail"]


def test_upload_skipped_without_boto3(tmp_path, monkeypatch):
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"fake model")

    # Simulate boto3 not installed
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("boto3 not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    from app.aws_stub import upload_model
    result = upload_model(model_file)
    assert result["status"] == "skipped"


def test_download_skipped_without_boto3(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("boto3 not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    from app.aws_stub import download_training_data
    result = download_training_data("data/train.csv", tmp_path / "train.csv")
    assert result["status"] == "skipped"
