"""Shared pytest fixtures for Energy-Oracle tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DB_URL = "sqlite:///./test_energy_oracle.db"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        TEST_DB_URL, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_engine):
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    session = TestingSessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_predict_payload():
    return {
        "zone": "residential",
        "hour": 18,
        "day_of_week": 1,
        "temperature": 25.0,
        "humidity": 60.0,
    }


@pytest.fixture
def commercial_payload():
    return {
        "zone": "commercial",
        "hour": 10,
        "day_of_week": 2,
        "temperature": 22.0,
        "humidity": 55.0,
    }


@pytest.fixture
def industrial_payload():
    return {
        "zone": "industrial",
        "hour": 14,
        "day_of_week": 3,
        "temperature": 28.0,
        "humidity": 45.0,
    }


@pytest.fixture
def reference_series():
    import numpy as np
    rng = np.random.default_rng(0)
    return rng.normal(50, 10, 200).tolist()


@pytest.fixture
def current_series_stable():
    import numpy as np
    rng = np.random.default_rng(1)
    return rng.normal(50, 10, 50).tolist()


@pytest.fixture
def current_series_drifted():
    import numpy as np
    rng = np.random.default_rng(2)
    return rng.normal(80, 15, 50).tolist()


@pytest.fixture
def all_zones_batch_payload():
    return [
        {"zone": z, "hour": 12, "day_of_week": 2, "temperature": 20.0, "humidity": 50.0}
        for z in ("residential", "commercial", "industrial", "mixed")
    ]
