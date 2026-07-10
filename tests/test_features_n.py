"""Tests for features module constants."""

from __future__ import annotations


def test_n_features_matches_feature_columns():
    from app.features import FEATURE_COLUMNS, N_FEATURES

    assert len(FEATURE_COLUMNS) == N_FEATURES
    assert N_FEATURES == 26


def test_feature_columns_unique():
    from app.features import FEATURE_COLUMNS

    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_zone_categories_count():
    from app.features import ZONE_CATEGORIES

    assert len(ZONE_CATEGORIES) == 4
    assert "residential" in ZONE_CATEGORIES
    assert "commercial" in ZONE_CATEGORIES
    assert "industrial" in ZONE_CATEGORIES
    assert "mixed" in ZONE_CATEGORIES


def test_encode_zone_handles_unknown_zone():
    import pandas as pd

    from app.features import encode_zone

    df = pd.DataFrame(
        {
            "zone": ["residential", "unknown_zone"],
            "hour": [0, 0],
            "day_of_week": [0, 0],
            "temperature": [20.0, 20.0],
            "humidity": [50.0, 50.0],
        }
    )
    result = encode_zone(df)
    assert "zone_encoded" in result.columns
