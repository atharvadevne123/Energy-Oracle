"""Simple file-based model registry for version tracking and rollback."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


__all__ = ['register_model', 'load_registry', 'get_production_model', 'promote', 'REGISTRY_PATH']
REGISTRY_PATH = Path("model_registry.json")


def register_model(
    version: str,
    metrics: dict[str, Any],
    model_path: str | Path,
    *,
    promoted: bool = False,
) -> dict[str, Any]:
    """
    Add a model version entry to the registry.

    Args:
        version: Semantic version string (e.g. '1.0.0').
        metrics: Training metrics dict.
        model_path: Path to the serialised model artifact.
        promoted: Whether this version is promoted to production.

    Returns:
        The registry entry created.
    """
    registry = load_registry()
    entry: dict[str, Any] = {
        "version": version,
        "model_path": str(model_path),
        "metrics": metrics,
        "promoted": promoted,
        "registered_at": datetime.now(tz=UTC).isoformat(),
    }
    registry.append(entry)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    logger.info("Registered model version=%s promoted=%s", version, promoted)
    return entry


def load_registry() -> list[dict[str, Any]]:
    """Load all registry entries from disk."""
    if not REGISTRY_PATH.exists():
        return []
    return json.loads(REGISTRY_PATH.read_text())


def get_production_model() -> dict[str, Any] | None:
    """Return the currently promoted production model entry, or None."""
    registry = load_registry()
    promoted = [r for r in registry if r.get("promoted")]
    return promoted[-1] if promoted else None


def promote(version: str) -> bool:
    """
    Promote a specific model version to production.

    Demotes all other versions first.

    Returns:
        True if the version was found and promoted.
    """
    registry = load_registry()
    found = False
    for entry in registry:
        if entry["version"] == version:
            entry["promoted"] = True
            found = True
        else:
            entry["promoted"] = False
    if found:
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
        logger.info("Promoted model version=%s", version)
    return found

