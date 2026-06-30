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

## [1.1.0] — 2026-06-30

### Added
- `/api/v1/version` endpoint returning app name, version, and model version
- `/api/v1/batch` endpoint for vectorized multi-record predictions (up to 1 000 records)
- `/api/v1/health/deep` endpoint with per-component database and model status
- `BatchPredictRequest`, `BatchPredictResponse`, `VersionResponse` Pydantic schemas
- LRU eviction for the in-memory prediction cache (`collections.OrderedDict`)
- Module-level LabelEncoder singleton in `features.py` (eliminates per-call re-fit overhead)
- Module-level model cache in `model.py` (single `joblib.load` per process lifetime)
- Composite database index on `(zone, created_at)` for faster log queries
- `CORS_ORIGINS` environment variable for configurable allowed origins
- `__all__` exports for every `app/` module
- `SECURITY.md` with vulnerability reporting guidance
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- `[tool.mypy]` configuration in `pyproject.toml`
- `type-check`, `coverage`, and `format` Makefile targets
- Extended pytest suite: `test_config`, `test_schemas`, `test_batch`, `test_report`, `test_retrain`, `test_middleware`

### Changed
- Batch prediction now builds a single vectorised `DataFrame` instead of per-row loops
- `CORS_ORIGINS` defaults to `["*"]` (unchanged behaviour, now env-configurable)
- All `datetime.utcnow()` calls replaced with `datetime.now(tz=UTC)` (deprecation fix)
- CI workflow upgraded to `actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`

### Fixed
- `load_model()` previously re-read `model.joblib` on every prediction request
- `encode_zone()` previously instantiated and fitted a new `LabelEncoder` on every call
