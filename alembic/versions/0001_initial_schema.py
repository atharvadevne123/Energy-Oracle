"""Initial schema — prediction_logs, training_runs, drift_events.

Revision ID: 0001
Revises:
Create Date: 2026-05-06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("correlation_id", sa.String(64), nullable=False, index=True),
        sa.Column("zone", sa.String(64), nullable=False),
        sa.Column("hour", sa.Integer, nullable=False),
        sa.Column("day_of_week", sa.Integer, nullable=False),
        sa.Column("temperature", sa.Float, nullable=False),
        sa.Column("humidity", sa.Float, nullable=False),
        sa.Column("predicted_kwh", sa.Float, nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False, server_default="1.0.0"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("run_id", sa.String(64), unique=True, nullable=False),
        sa.Column("auc_mean", sa.Float, nullable=True),
        sa.Column("auc_std", sa.Float, nullable=True),
        sa.Column("rmse", sa.Float, nullable=True),
        sa.Column("n_samples", sa.Integer, nullable=False),
        sa.Column("n_features", sa.Integer, nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "drift_events",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("feature", sa.String(64), nullable=False),
        sa.Column("ks_statistic", sa.Float, nullable=False),
        sa.Column("p_value", sa.Float, nullable=False),
        sa.Column("drift_detected", sa.Integer, nullable=False),
        sa.Column("window_size", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("drift_events")
    op.drop_table("training_runs")
    op.drop_table("prediction_logs")
