"""FastAPI application — /api/v1/predict, /api/v1/health, /api/v1/metrics, /api/v1/drift."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import single_row_to_df
from app.model import load_metrics, predict
from app.monitoring import log_prediction, run_drift_check, summarise_predictions

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# Simple in-memory rate limiter: tracks request timestamps per IP
_rate_store: dict[str, list[float]] = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Energy-Oracle API started")
    yield
    logger.info("Energy-Oracle API shutting down")


app = FastAPI(
    title="Energy-Oracle",
    description=(
        "Real-time energy consumption prediction API using LightGBM ensemble "
        "with time-series feature engineering and drift detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a correlation ID to every request for distributed tracing."""
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.correlation_id = corr_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple sliding-window rate limiter (per client IP)."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _rate_store.setdefault(client_ip, [])
    _rate_store[client_ip] = [t for t in window if now - t < 60]
    if len(_rate_store[client_ip]) >= RATE_LIMIT:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {"detail": "Rate limit exceeded. Max 60 requests/minute."},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    _rate_store[client_ip].append(now)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Input schema for energy consumption prediction."""

    zone: str = Field(
        ...,
        description="Zone type: residential | commercial | industrial | mixed",
        examples=["residential"],
    )
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    temperature: float = Field(..., ge=-20.0, le=50.0, description="Temperature in °C")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity (%)")

    @field_validator("zone")
    @classmethod
    def validate_zone(cls, v: str) -> str:
        allowed = {"residential", "commercial", "industrial", "mixed"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"zone must be one of {sorted(allowed)}")
        return v


class PredictResponse(BaseModel):
    """Prediction output."""

    predicted_kwh: float = Field(..., description="Predicted energy consumption in kWh")
    zone: str
    hour: int
    day_of_week: int
    model_version: str
    correlation_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    training_metrics: dict[str, Any]
    prediction_summary: dict[str, Any]


_start_time = time.time()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
def health() -> HealthResponse:
    """Return service health status and uptime."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    summary="Predict energy consumption",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
def predict_endpoint(
    body: PredictRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> PredictResponse:
    """
    Predict energy consumption (kWh) for a zone at a given hour.

    Uses a blended LightGBM + RandomForest ensemble with time-series features.
    Every prediction is logged to the database for monitoring.
    """
    corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    try:
        features = single_row_to_df(body.model_dump())
        kwh = predict(features)
    except Exception as exc:
        logger.error("Prediction failed corr=%s: %s", corr_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check server logs.",
        ) from exc

    log_prediction(
        db,
        correlation_id=corr_id,
        zone=body.zone,
        hour=body.hour,
        day_of_week=body.day_of_week,
        temperature=body.temperature,
        humidity=body.humidity,
        predicted_kwh=kwh,
    )

    return PredictResponse(
        predicted_kwh=kwh,
        zone=body.zone,
        hour=body.hour,
        day_of_week=body.day_of_week,
        model_version="1.0.0",
        correlation_id=corr_id,
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    summary="Model and prediction metrics",
    tags=["Monitoring"],
)
def metrics_endpoint(db: Annotated[Session, Depends(get_db)]) -> MetricsResponse:
    """Return training metrics and a summary of recent predictions."""
    return MetricsResponse(
        training_metrics=load_metrics(),
        prediction_summary=summarise_predictions(db),
    )


@app.get(
    "/api/v1/drift",
    summary="Run drift detection",
    tags=["Monitoring"],
)
def drift_endpoint(db: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    """
    Run KS-test drift detection across temperature, humidity, and predicted_kwh.

    Requires at least 60 logged predictions to produce meaningful results.
    """
    return run_drift_check(db)
