"""Data quality checks for incoming prediction requests and training data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


__all__ = ["check_missing_values", "check_outliers", "validate_training_dataframe"]
EXPECTED_DTYPES: dict[str, type] = {
    "zone": str,
    "hour": int,
    "day_of_week": int,
    "temperature": float,
    "humidity": float,
}


def check_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return count of missing values per column."""
    return df.isnull().sum().to_dict()


def check_outliers(df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Identify outlier row indices for numeric columns using IQR method.

    Returns:
        Dict mapping column name to list of outlier row indices.
    """
    outliers: dict[str, list[int]] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        idx = df.index[(df[col] < lower) | (df[col] > upper)].tolist()
        if idx:
            outliers[col] = idx
    return outliers


def validate_training_dataframe(df: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """
    Run data quality checks on a training DataFrame.

    Args:
        df: Feature DataFrame.
        target: Target Series.

    Returns:
        Dict with quality metrics and warnings.
    """
    warnings: list[str] = []
    missing = check_missing_values(df)
    total_missing = sum(missing.values())
    if total_missing > 0:
        warnings.append(
            f"{total_missing} missing values found across {sum(1 for v in missing.values() if v > 0)} columns."
        )

    if len(df) != len(target):
        warnings.append(f"Feature rows ({len(df)}) != target rows ({len(target)}).")

    if (target < 0).any():
        warnings.append("Target contains negative kWh values.")

    outliers = check_outliers(df.select_dtypes(include=[np.number]))
    n_outlier_rows = sum(len(v) for v in outliers.values())

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing_values": missing,
        "total_missing": total_missing,
        "outlier_rows": n_outlier_rows,
        "outlier_columns": list(outliers.keys()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": round(float(target.mean()), 4),
        "warnings": warnings,
        "quality_ok": len(warnings) == 0,
    }
