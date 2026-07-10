"""Tests for feature engineering pipeline."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "zone": ["residential", "commercial", "industrial", "mixed"],
            "hour": [0, 8, 17, 22],
            "day_of_week": [0, 3, 5, 6],
            "temperature": [10.0, 20.0, 30.0, 15.0],
            "humidity": [40.0, 60.0, 80.0, 50.0],
        }
    )


def test_add_time_features_creates_expected_columns(sample_df):
    from app.features import add_time_features

    result = add_time_features(sample_df)
    for col in [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "is_peak_hour",
        "is_off_peak",
    ]:
        assert col in result.columns


def test_is_weekend_correct(sample_df):
    from app.features import add_time_features

    result = add_time_features(sample_df)
    # days 5,6 are weekend
    assert result.loc[result["day_of_week"] == 5, "is_weekend"].iloc[0] == 1
    assert result.loc[result["day_of_week"] == 0, "is_weekend"].iloc[0] == 0


def test_is_peak_hour_correct(sample_df):
    from app.features import add_time_features

    result = add_time_features(sample_df)
    assert result.loc[result["hour"] == 17, "is_peak_hour"].iloc[0] == 1
    assert result.loc[result["hour"] == 0, "is_peak_hour"].iloc[0] == 0


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_lag_features_created(sample_df, lag):
    from app.features import add_lag_features

    result = add_lag_features(sample_df)
    assert f"temp_lag_{lag}" in result.columns
    assert f"humidity_lag_{lag}" in result.columns


@pytest.mark.parametrize("window", [3, 6, 12])
def test_rolling_features_created(sample_df, window):
    from app.features import add_rolling_features

    result = add_rolling_features(sample_df)
    assert f"temp_rolling_mean_{window}" in result.columns
    assert f"temp_rolling_std_{window}" in result.columns


def test_ratio_features_heat_index(sample_df):
    from app.features import add_ratio_features

    result = add_ratio_features(sample_df)
    assert "heat_index" in result.columns
    assert "temp_humidity_ratio" in result.columns
    assert "temp_squared" in result.columns


def test_encode_zone_values(sample_df):
    from app.features import encode_zone

    result = encode_zone(sample_df)
    assert "zone_encoded" in result.columns
    assert result["zone_encoded"].between(0, 3).all()


def test_build_feature_matrix_columns(sample_df):
    from app.features import FEATURE_COLUMNS, build_feature_matrix

    result = build_feature_matrix(sample_df)
    assert list(result.columns) == FEATURE_COLUMNS


def test_single_row_to_df_shape():
    from app.features import FEATURE_COLUMNS, single_row_to_df

    result = single_row_to_df(
        {
            "zone": "industrial",
            "hour": 15,
            "day_of_week": 2,
            "temperature": 28.0,
            "humidity": 65.0,
        }
    )
    assert result.shape == (1, len(FEATURE_COLUMNS))


def test_feature_matrix_no_nan(sample_df):
    from app.features import build_feature_matrix

    result = build_feature_matrix(sample_df)
    assert not result.isnull().any().any(), "Feature matrix contains NaN values"
