"""LightGBM ensemble model training, evaluation, and inference."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features import FEATURE_COLUMNS, build_feature_matrix

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
MODEL_VERSION = "1.0.0"

__all__ = ["generate_synthetic_data", "train_model", "load_model", "predict", "load_metrics", "MODEL_PATH", "MODEL_VERSION"]

_model_cache: dict[str, Pipeline] | None = None


def _build_pipeline(model: Any) -> Pipeline:
    """Wrap a regressor in a scaling pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _make_lgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def _make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )


def generate_synthetic_data(n_samples: int = 5000) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate realistic synthetic energy consumption data.

    Consumption is driven by: zone baseline + temperature effect +
    peak-hour surge + random noise.
    """
    rng = np.random.default_rng(42)
    zones = rng.choice(["residential", "commercial", "industrial", "mixed"], n_samples)
    hours = rng.integers(0, 24, n_samples)
    days = rng.integers(0, 7, n_samples)
    temps = rng.normal(20, 8, n_samples).clip(-5, 40)
    humidity = rng.uniform(30, 90, n_samples)

    zone_base = {"residential": 25.0, "commercial": 80.0, "industrial": 250.0, "mixed": 60.0}
    base = np.array([zone_base[z] for z in zones])
    temp_effect = 0.8 * np.maximum(temps - 18, 0) + 0.5 * np.maximum(8 - temps, 0)
    peak_surge = np.where(np.isin(hours, range(17, 22)), base * 0.3, 0.0)
    weekend_drop = np.where(np.isin(days, [5, 6]), -base * 0.15, 0.0)
    noise = rng.normal(0, base * 0.05)
    kwh = (base + temp_effect + peak_surge + weekend_drop + noise).clip(min=0.5)

    df = pd.DataFrame(
        {"zone": zones, "hour": hours, "day_of_week": days,
         "temperature": temps, "humidity": humidity}
    )
    target = pd.Series(kwh, name="kwh")
    return df, target


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train LightGBM ensemble with 5-fold CV and persist model + metrics."""
    run_id = str(uuid.uuid4())
    logger.info("Starting training run %s — %d samples", run_id, len(X))

    X_feat = build_feature_matrix(X)

    lgbm_pipe = _build_pipeline(_make_lgbm())
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    neg_rmse_scores = cross_val_score(
        lgbm_pipe, X_feat, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    rmse_scores = -neg_rmse_scores

    # Fit final model on full data
    lgbm_pipe.fit(X_feat, y)

    # Secondary RF ensemble for blending
    rf_pipe = _build_pipeline(_make_rf())
    rf_pipe.fit(X_feat, y)

    metrics = {
        "run_id": run_id,
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "n_samples": len(X),
        "n_features": len(FEATURE_COLUMNS),
        "model_version": MODEL_VERSION,
        "cv_folds": cv_folds,
    }

    joblib.dump({"lgbm": lgbm_pipe, "rf": rf_pipe}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    global _model_cache
    _model_cache = {"lgbm": lgbm_pipe, "rf": rf_pipe}

    logger.info(
        "Training complete — RMSE %.4f ± %.4f", metrics["rmse_mean"], metrics["rmse_std"]
    )
    return lgbm_pipe, metrics


def load_model() -> dict[str, Pipeline]:
    """Return the in-process model cache, loading from disk only on first call."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not MODEL_PATH.exists():
        logger.info("No model found — training on synthetic data")
        df, y = generate_synthetic_data()
        train_model(df, y)
        return _model_cache  # type: ignore[return-value]
    _model_cache = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    return _model_cache


def predict(features: pd.DataFrame, blend_alpha: float = 0.7) -> float:
    """
    Return blended ensemble prediction.

    blend_alpha controls LightGBM weight (1-alpha goes to RandomForest).
    """
    ensemble = load_model()
    lgbm_pred = float(ensemble["lgbm"].predict(features)[0])
    rf_pred = float(ensemble["rf"].predict(features)[0])
    blended = blend_alpha * lgbm_pred + (1 - blend_alpha) * rf_pred
    return max(0.0, round(blended, 4))


def load_metrics() -> dict[str, Any]:
    """Return stored training metrics, or empty dict if unavailable."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}
