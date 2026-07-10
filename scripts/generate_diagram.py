"""Generate Energy-Oracle system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)


def draw_box(ax, x, y, w, h, label, color, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        zorder=3,
    )


def draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.5},
        zorder=1,
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    plt.title(
        "Energy-Oracle — System Architecture",
        fontsize=16,
        fontweight="bold",
        pad=15,
        color="#1a1a2e",
    )

    # Client
    draw_box(ax, 0.3, 3.8, 2.0, 1.2, "Client\n(HTTP)", "#4A90E2", 9)

    # API Gateway / FastAPI
    draw_box(
        ax, 3.2, 3.5, 2.8, 1.8, "FastAPI\n/api/v1/predict\n/health  /metrics  /drift", "#2ECC71", 8
    )

    # Feature Engineering
    draw_box(
        ax, 7.0, 5.5, 2.8, 1.4, "Feature Engineering\n(lag · rolling · cyclical)", "#E67E22", 8
    )

    # ML Ensemble
    draw_box(ax, 7.0, 3.5, 2.8, 1.6, "ML Ensemble\nLightGBM 70%\nRandom Forest 30%", "#9B59B6", 8)

    # Model Monitoring
    draw_box(ax, 7.0, 1.5, 2.8, 1.4, "Monitoring\nKS-test drift\nPrediction logging", "#E74C3C", 8)

    # PostgreSQL
    draw_box(
        ax, 11.0, 3.8, 2.8, 1.2, "PostgreSQL\nPredictions · Drift\nTraining runs", "#16A085", 8
    )

    # Retraining DAG
    draw_box(ax, 11.0, 6.0, 2.8, 1.2, "Airflow DAG\nWeekly retrain\n& validation", "#F39C12", 8)

    # Model Store
    draw_box(ax, 7.0, 7.2, 2.8, 1.2, "Model Store\nmodel.joblib\nmetrics.json", "#1ABC9C", 8)

    # Arrows
    draw_arrow(ax, 2.3, 4.4, 3.2, 4.4)  # Client -> FastAPI
    draw_arrow(ax, 6.0, 4.8, 7.0, 6.0)  # FastAPI -> Feature Eng
    draw_arrow(ax, 8.4, 5.5, 8.4, 5.1)  # Feature Eng -> ML Ensemble
    draw_arrow(ax, 6.0, 4.3, 7.0, 4.3)  # FastAPI -> ML Ensemble
    draw_arrow(ax, 6.0, 3.8, 7.0, 2.2)  # FastAPI -> Monitoring
    draw_arrow(ax, 9.8, 4.3, 11.0, 4.4)  # ML Ensemble -> Postgres
    draw_arrow(ax, 9.8, 2.2, 11.0, 4.0)  # Monitoring -> Postgres
    draw_arrow(ax, 11.0, 6.6, 9.8, 7.8)  # Airflow -> Model Store
    draw_arrow(ax, 8.4, 7.2, 8.4, 5.1)  # Model Store -> ML Ensemble

    # Legend
    legend_items = [
        mpatches.Patch(color="#4A90E2", label="Client"),
        mpatches.Patch(color="#2ECC71", label="FastAPI"),
        mpatches.Patch(color="#E67E22", label="Features"),
        mpatches.Patch(color="#9B59B6", label="ML Ensemble"),
        mpatches.Patch(color="#E74C3C", label="Monitoring"),
        mpatches.Patch(color="#16A085", label="Database"),
        mpatches.Patch(color="#F39C12", label="Airflow DAG"),
        mpatches.Patch(color="#1ABC9C", label="Model Store"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
        ncol=4,
    )

    plt.tight_layout()
    plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
    print("Architecture diagram saved to screenshots/architecture.png")


if __name__ == "__main__":
    main()
