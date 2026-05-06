"""Feature importance and explainability utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_feature_importances(model_ensemble: dict[str, Any]) -> dict[str, float]:
    """
    Extract normalised feature importances from the LightGBM component.

    Args:
        model_ensemble: Dict with 'lgbm' and 'rf' Pipeline objects.

    Returns:
        Dict mapping feature name to normalised importance (sums to 1.0).
    """
    from app.features import FEATURE_COLUMNS

    lgbm_pipe = model_ensemble.get("lgbm")
    if lgbm_pipe is None:
        return {}

    try:
        lgbm_model = lgbm_pipe.named_steps["model"]
        raw = lgbm_model.feature_importances_
        total = raw.sum()
        normalised = (raw / total).tolist() if total > 0 else raw.tolist()
        return dict(zip(FEATURE_COLUMNS, normalised, strict=False))
    except Exception as exc:
        logger.warning("Could not extract feature importances: %s", exc)
        return {}


def top_k_features(importances: dict[str, float], k: int = 10) -> list[tuple[str, float]]:
    """
    Return the top-k most important features sorted descending.

    Args:
        importances: Feature name → normalised importance mapping.
        k: Number of top features to return.

    Returns:
        List of (feature_name, importance) tuples.
    """
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:k]


def explain_prediction(
    features: pd.DataFrame,
    importances: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Produce a simple contribution breakdown for a single prediction.

    Args:
        features: Single-row feature DataFrame.
        importances: Normalised feature importances.

    Returns:
        List of dicts with feature, value, and importance.
    """
    row = features.iloc[0]
    breakdown = []
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
        val = row.get(feat, float("nan"))
        breakdown.append({
            "feature": feat,
            "value": round(float(val), 4) if not np.isnan(float(val)) else None,
            "importance": round(imp, 4),
        })
    return breakdown
