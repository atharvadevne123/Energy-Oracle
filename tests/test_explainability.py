"""Tests for explainability utilities."""

from __future__ import annotations


def test_get_feature_importances_returns_dict(tmp_path, monkeypatch):
    from app import model as m
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.explainability import get_feature_importances
    from app.model import generate_synthetic_data, load_model, train_model

    df, y = generate_synthetic_data(n_samples=300)
    train_model(df, y, cv_folds=2)
    ensemble = load_model()
    importances = get_feature_importances(ensemble)
    assert isinstance(importances, dict)
    assert len(importances) > 0


def test_importances_sum_to_one(tmp_path, monkeypatch):
    from app import model as m
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.explainability import get_feature_importances
    from app.model import generate_synthetic_data, load_model, train_model

    df, y = generate_synthetic_data(n_samples=300)
    train_model(df, y, cv_folds=2)
    ensemble = load_model()
    importances = get_feature_importances(ensemble)
    total = sum(importances.values())
    assert abs(total - 1.0) < 1e-6


def test_top_k_features_length():
    from app.explainability import top_k_features

    importances = {f"feat_{i}": float(i) / 100 for i in range(30)}
    top = top_k_features(importances, k=5)
    assert len(top) == 5
    # Should be sorted descending
    vals = [v for _, v in top]
    assert vals == sorted(vals, reverse=True)


def test_explain_prediction_structure(tmp_path, monkeypatch):
    from app import model as m
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    from app.explainability import explain_prediction, get_feature_importances
    from app.features import single_row_to_df
    from app.model import generate_synthetic_data, load_model, train_model

    df, y = generate_synthetic_data(n_samples=300)
    train_model(df, y, cv_folds=2)
    ensemble = load_model()
    importances = get_feature_importances(ensemble)
    features = single_row_to_df({"zone": "commercial", "hour": 14, "day_of_week": 3, "temperature": 22.0, "humidity": 60.0})
    breakdown = explain_prediction(features, importances)
    assert len(breakdown) > 0
    for item in breakdown:
        assert "feature" in item
        assert "importance" in item
