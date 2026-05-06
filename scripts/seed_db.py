"""Seed the database with synthetic predictions for testing drift detection."""

from __future__ import annotations

import argparse
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed(n: int = 200) -> None:
    """
    Insert n synthetic prediction records into the database.

    Args:
        n: Number of records to insert.
    """
    import numpy as np

    from app.database import SessionLocal, init_db
    from app.monitoring import log_prediction

    init_db()
    rng = np.random.default_rng(99)
    db = SessionLocal()

    zones = ["residential", "commercial", "industrial", "mixed"]
    try:
        for _ in range(n):
            log_prediction(
                db,
                correlation_id=str(uuid.uuid4()),
                zone=rng.choice(zones),
                hour=int(rng.integers(0, 24)),
                day_of_week=int(rng.integers(0, 7)),
                temperature=float(rng.normal(20, 8)),
                humidity=float(rng.uniform(30, 90)),
                predicted_kwh=float(rng.uniform(10, 300)),
            )
        logger.info("Seeded %d prediction records", n)
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Energy-Oracle database")
    parser.add_argument("--n", type=int, default=200, help="Number of records to insert")
    args = parser.parse_args()
    seed(args.n)
