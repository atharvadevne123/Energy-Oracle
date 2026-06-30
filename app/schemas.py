"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Input schema for energy consumption prediction endpoint."""

    zone: str = Field(
        ...,
        description="Zone type: residential | commercial | industrial | mixed",
        examples=["residential"],
    )
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    temperature: float = Field(..., ge=-20.0, le=50.0, description="Ambient temperature in °C")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity (0–100 %)")

    @field_validator("zone")
    @classmethod
    def validate_zone(cls, v: str) -> str:
        """Normalise and validate zone identifier."""
        allowed = {"residential", "commercial", "industrial", "mixed"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"zone must be one of {sorted(allowed)}")
        return v


class PredictResponse(BaseModel):
    """Prediction output schema."""

    predicted_kwh: float = Field(..., description="Predicted energy consumption in kWh")
    zone: str = Field(..., description="Zone used for prediction")
    hour: int = Field(..., description="Hour of day used for prediction")
    day_of_week: int = Field(..., description="Day-of-week used for prediction")
    model_version: str = Field(..., description="Version of the serving model")
    correlation_id: str = Field(..., description="Trace ID for this request")


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(..., description="'ok' when healthy")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Seconds since process start")


class MetricsResponse(BaseModel):
    """Training and operational metrics response."""

    training_metrics: dict[str, Any] = Field(..., description="Last training run metrics")
    prediction_summary: dict[str, Any] = Field(..., description="Summary of recent predictions")


class DriftResponse(BaseModel):
    """Drift detection response."""

    status: str = Field(..., description="'stable' | 'drift_detected' | 'insufficient_data'")
    features: dict[str, Any] | None = Field(None, description="Per-feature KS-test results")
    reference_window: int | None = Field(None, description="Number of reference samples")
    current_window: int | None = Field(None, description="Number of current samples")


class BatchPredictItem(BaseModel):
    """Single record in a batch prediction request."""

    zone: str = Field(..., description="Zone type: residential | commercial | industrial | mixed")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    temperature: float = Field(..., ge=-20.0, le=50.0, description="Ambient temperature in °C")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity (0–100 %)")

    @field_validator("zone")
    @classmethod
    def validate_zone(cls, v: str) -> str:
        """Normalise and validate zone identifier."""
        allowed = {"residential", "commercial", "industrial", "mixed"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"zone must be one of {sorted(allowed)}")
        return v


class BatchPredictRequest(BaseModel):
    """Input schema for bulk prediction endpoint."""

    records: list[BatchPredictItem] = Field(..., description="List of prediction inputs (max 1000)")


class BatchPredictResultItem(BaseModel):
    """Result for a single record in a batch prediction response."""

    predicted_kwh: float | None = Field(None, description="Predicted kWh, None if record errored")
    error: str | None = Field(None, description="Error message if prediction failed")


class BatchPredictResponse(BaseModel):
    """Bulk prediction output schema."""

    results: list[dict[str, Any]] = Field(..., description="Per-record results with predicted_kwh")
    total: int = Field(..., description="Total records submitted")
    successful: int = Field(..., description="Records successfully predicted")


class VersionResponse(BaseModel):
    """Application version metadata."""

    name: str = Field(..., description="Application name")
    version: str = Field(..., description="Semantic version")
    model_version: str = Field(..., description="Currently loaded model version")
