"""Tests for model training and prediction."""

from __future__ import annotations

import pytest


def test_generate_synthetic_data_shape():
    from app.model import generate_synthetic_data

    df, y = generate_synthetic_data(n_samples=200)
    assert len(df) == 200
    assert len(y) == 200
    assert set(df.columns) >= {"zone", "hour", "day_of_week", "temperature", "humidity"}


def test_generate_synthetic_data_value_ranges():
    from app.model import generate_synthetic_data

    df, y = generate_synthetic_data(n_samples=500)
    assert df["hour"].between(0, 23).all()
    assert df["day_of_week"].between(0, 6).all()
    assert y.min() > 0


@pytest.mark.parametrize("n_samples", [100, 500, 1000])
def test_train_model_returns_metrics(n_samples, tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.model import generate_synthetic_data, train_model

    df, y = generate_synthetic_data(n_samples=n_samples)
    _, metrics = train_model(df, y, cv_folds=2)
    assert "rmse_mean" in metrics
    assert metrics["rmse_mean"] >= 0
    assert metrics["n_features"] > 0


def test_predict_returns_positive_float(tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.features import single_row_to_df
    from app.model import generate_synthetic_data, train_model

    df, y = generate_synthetic_data(n_samples=200)
    train_model(df, y, cv_folds=2)

    features = single_row_to_df(
        {
            "zone": "commercial",
            "hour": 14,
            "day_of_week": 3,
            "temperature": 22.0,
            "humidity": 60.0,
        }
    )
    from app.model import predict

    result = predict(features)
    assert isinstance(result, float)
    assert result > 0


def test_load_metrics_returns_dict(tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.model import load_metrics

    assert isinstance(load_metrics(), dict)


def test_blend_alpha_affects_prediction(tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.features import single_row_to_df
    from app.model import generate_synthetic_data, train_model

    df, y = generate_synthetic_data(n_samples=200)
    train_model(df, y, cv_folds=2)

    features = single_row_to_df(
        {
            "zone": "residential",
            "hour": 20,
            "day_of_week": 1,
            "temperature": 30.0,
            "humidity": 70.0,
        }
    )
    from app.model import predict

    # Different blend alphas may yield different results — both should be positive
    p1 = predict(features, blend_alpha=0.9)
    p2 = predict(features, blend_alpha=0.1)
    assert p1 > 0 and p2 > 0
