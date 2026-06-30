"""Feature engineering pipeline for energy consumption prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

ZONE_CATEGORIES = ["residential", "commercial", "industrial", "mixed"]

# Pre-fitted LabelEncoder singleton — avoids re-instantiation on every inference call
_ZONE_ENCODER: LabelEncoder = LabelEncoder()
_ZONE_ENCODER.fit(ZONE_CATEGORIES)

# Baseline consumption per zone (kWh reference for ratio features)
ZONE_BASELINES: dict[str, float] = {
    "residential": 25.0,
    "commercial": 80.0,
    "industrial": 250.0,
    "mixed": 60.0,
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour and day-of-week encodings using sin/cos transforms."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin(range(17, 22)).astype(int)
    df["is_off_peak"] = df["hour"].isin(range(0, 6)).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Add lagged temperature and humidity to capture autocorrelation."""
    if lags is None:
        lags = [1, 2, 3]
    df = df.copy()
    for lag in lags:
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag).fillna(df["temperature"].mean())
        df[f"humidity_lag_{lag}"] = df["humidity"].shift(lag).fillna(df["humidity"].mean())
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling mean and std for temperature over sliding windows."""
    if windows is None:
        windows = [3, 6, 12]
    df = df.copy()
    for w in windows:
        df[f"temp_rolling_mean_{w}"] = (
            df["temperature"].rolling(w, min_periods=1).mean()
        )
        df[f"temp_rolling_std_{w}"] = (
            df["temperature"].rolling(w, min_periods=1).std().fillna(0)
        )
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature-humidity interaction and zone-relative ratios."""
    df = df.copy()
    df["heat_index"] = df["temperature"] * (1 + 0.018 * (df["humidity"] - 50))
    df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"].clip(lower=1))
    df["temp_squared"] = df["temperature"] ** 2
    # Zone normalised temperature
    df["zone_baseline"] = df["zone"].map(ZONE_BASELINES).fillna(60.0)
    df["temp_per_baseline"] = df["temperature"] / df["zone_baseline"]
    return df


def encode_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode the zone column using the pre-fitted module-level encoder."""
    df = df.copy()
    df["zone_encoded"] = _ZONE_ENCODER.transform(
        df["zone"].where(df["zone"].isin(ZONE_CATEGORIES), other="mixed")
    )
    return df


FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_peak_hour", "is_off_peak",
    "temperature", "humidity",
    "heat_index", "temp_humidity_ratio", "temp_squared",
    "temp_per_baseline", "zone_encoded",
    "temp_lag_1", "temp_lag_2", "temp_lag_3",
    "humidity_lag_1", "humidity_lag_2", "humidity_lag_3",
    "temp_rolling_mean_3", "temp_rolling_std_3",
    "temp_rolling_mean_6", "temp_rolling_std_6",
    "temp_rolling_mean_12", "temp_rolling_std_12",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering pipeline and return feature matrix."""
    logger.debug("Building feature matrix from %d rows", len(df))
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ratio_features(df)
    df = encode_zone(df)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after engineering: {missing}")
    result = df[FEATURE_COLUMNS].copy()
    logger.debug("Feature matrix shape: %s", result.shape)
    return result


def build_sklearn_pipeline() -> Pipeline:
    """Return a sklearn Pipeline that scales features (used in CV)."""
    return Pipeline([("scaler", StandardScaler())])


def single_row_to_df(data: dict[str, Any]) -> pd.DataFrame:
    """Convert a single prediction request dict to a feature-ready DataFrame."""
    row = {
        "zone": data["zone"],
        "hour": int(data["hour"]),
        "day_of_week": int(data["day_of_week"]),
        "temperature": float(data["temperature"]),
        "humidity": float(data["humidity"]),
    }
    df = pd.DataFrame([row])
    return build_feature_matrix(df)
