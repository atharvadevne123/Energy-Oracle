"""Tests for data quality validation module."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def clean_df():
    return pd.DataFrame(
        {
            "temperature": [20.0, 25.0, 15.0, 30.0, 18.0],
            "humidity": [50.0, 60.0, 70.0, 45.0, 55.0],
        }
    )


@pytest.fixture
def clean_target():
    return pd.Series([25.0, 30.0, 20.0, 35.0, 28.0])


def test_no_missing_values(clean_df):
    from app.data_quality import check_missing_values

    missing = check_missing_values(clean_df)
    assert all(v == 0 for v in missing.values())


def test_missing_values_detected():
    from app.data_quality import check_missing_values

    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, None, 1.0]})
    missing = check_missing_values(df)
    assert missing["a"] == 1
    assert missing["b"] == 2


def test_outliers_detected():
    from app.data_quality import check_outliers

    # 9 normal values + 1 extreme outlier
    df = pd.DataFrame({"val": [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.2, 11.2, 12.2, 1000.0]})
    outliers = check_outliers(df)
    assert "val" in outliers
    assert 9 in outliers["val"]  # last index is the outlier


def test_validate_clean_data_passes(clean_df, clean_target):
    from app.data_quality import validate_training_dataframe

    result = validate_training_dataframe(clean_df, clean_target)
    assert result["quality_ok"]
    assert result["warnings"] == []
    assert result["n_rows"] == 5


def test_validate_with_negative_target_warns(clean_df):
    from app.data_quality import validate_training_dataframe

    target = pd.Series([-1.0, 10.0, 20.0, 30.0, 40.0])
    result = validate_training_dataframe(clean_df, target)
    assert not result["quality_ok"]
    assert any("negative" in w for w in result["warnings"])


def test_validate_mismatched_rows_warns(clean_df):
    from app.data_quality import validate_training_dataframe

    target = pd.Series([1.0, 2.0])  # too short
    result = validate_training_dataframe(clean_df, target)
    assert not result["quality_ok"]
