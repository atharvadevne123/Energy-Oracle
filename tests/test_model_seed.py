"""Tests for generate_synthetic_data seed reproducibility."""

from __future__ import annotations


def test_same_seed_produces_identical_data():
    from app.model import generate_synthetic_data

    df1, y1 = generate_synthetic_data(n_samples=100, seed=7)
    df2, y2 = generate_synthetic_data(n_samples=100, seed=7)
    assert list(df1["hour"]) == list(df2["hour"])
    assert list(y1) == list(y2)


def test_different_seeds_produce_different_data():
    from app.model import generate_synthetic_data

    _, y1 = generate_synthetic_data(n_samples=100, seed=1)
    _, y2 = generate_synthetic_data(n_samples=100, seed=2)
    assert list(y1) != list(y2)


def test_synthetic_data_shape(tmp_path, monkeypatch):
    from app.model import generate_synthetic_data

    df, y = generate_synthetic_data(n_samples=300, seed=42)
    assert len(df) == 300
    assert len(y) == 300
    assert set(df.columns) >= {"zone", "hour", "day_of_week", "temperature", "humidity"}


def test_synthetic_data_zones_valid():
    from app.model import generate_synthetic_data

    df, _ = generate_synthetic_data(n_samples=200, seed=0)
    valid_zones = {"residential", "commercial", "industrial", "mixed"}
    assert set(df["zone"].unique()).issubset(valid_zones)


def test_synthetic_data_kwh_positive():
    from app.model import generate_synthetic_data

    _, y = generate_synthetic_data(n_samples=200, seed=0)
    assert (y > 0).all()
