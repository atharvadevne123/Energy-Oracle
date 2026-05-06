# Changelog

All notable changes to Energy-Oracle are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2026-05-06

### Added
- LightGBM + RandomForest blended ensemble with 5-fold cross-validation
- 26-feature time-series pipeline: cyclical encodings, lag features, rolling stats, ratio features
- FastAPI `/api/v1/predict`, `/api/v1/health`, `/api/v1/metrics`, `/api/v1/drift` endpoints
- KS-test drift detection across temperature, humidity, and predicted_kwh distributions
- SQLAlchemy models for prediction logs, training runs, and drift events
- Docker + docker-compose with PostgreSQL support
- Automated retraining DAG (Airflow-compatible, standalone fallback)
- Correlation ID middleware for distributed tracing
- Sliding-window rate limiter (60 req/min per IP)
- GitHub Actions CI with Ruff lint + pytest
- Architecture diagram (matplotlib)
- Alembic migration setup
- Full pytest suite with parametrized tests and fixtures
