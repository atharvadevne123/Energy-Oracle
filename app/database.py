"""SQLAlchemy models and session management for Energy-Oracle."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

__all__ = ["Base", "PredictionLog", "TrainingRun", "DriftEvent", "SessionLocal", "engine", "init_db", "get_db"]

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./energy_oracle.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Stores every prediction for monitoring and drift detection."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    correlation_id = Column(String(64), index=True, nullable=False)
    zone = Column(String(64), nullable=False)
    hour = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    predicted_kwh = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False, default="1.0.0")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    __table_args__ = (
        Index("ix_prediction_logs_zone_created_at", "zone", "created_at"),
    )


class TrainingRun(Base):
    """Records each model training event with metrics."""

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, nullable=False)
    auc_mean = Column(Float, nullable=True)
    auc_std = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=False)
    n_features = Column(Integer, nullable=False)
    model_version = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


class DriftEvent(Base):
    """Records drift detection results."""

    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, index=True)
    feature = Column(String(64), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False)  # 0/1 boolean
    window_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", DATABASE_URL)


def get_db() -> Session:
    """Yield a database session, closing it on completion."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
