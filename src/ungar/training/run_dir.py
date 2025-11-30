"""Run directory and manifest management."""

from __future__ import annotations

import datetime
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

ANALYTICS_SCHEMA_VERSION = 1


@dataclass
class RunManifest:
    """Metadata for a training run."""

    run_id: str
    timestamp: float
    created_at: str
    analytics_schema_version: int
    game: str
    algo: str
    config: Dict[str, Any]
    device: str
    metrics_path: str
    overlays_path: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunManifest:
        return cls(**data)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class RunPaths:
    """Paths to artifacts in a run directory."""

    root: Path
    manifest: Path
    metrics: Path
    overlays: Path
    config: Path


def create_run_dir(
    game: str,
    algo: str,
    config_dict: Dict[str, Any],
    device: str,
    base_dir: str | Path | None = None,
    run_id: str | None = None,
    notes: str | None = None,
) -> RunPaths:
    """Initialize a standardized run directory."""
    if base_dir is None:
        base_dir = Path("runs")
    else:
        base_dir = Path(base_dir)

    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    timestamp = time.time()
    created_at = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).isoformat()

    # Format: <timestamp_int>_<game>_<algo>_<short_id>
    dirname = f"{int(timestamp)}_{game}_{algo}_{run_id}"
    run_root = base_dir / dirname

    run_root.mkdir(parents=True, exist_ok=True)
    overlays_dir = run_root / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    manifest_path = run_root / "manifest.json"
    metrics_path = run_root / "metrics.csv"
    config_path = run_root / "config.json"

    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Create and save manifest
    manifest = RunManifest(
        run_id=run_id,
        timestamp=timestamp,
        created_at=created_at,
        analytics_schema_version=ANALYTICS_SCHEMA_VERSION,
        game=game,
        algo=algo,
        config=config_dict,
        device=device,
        metrics_path=metrics_path.name,
        overlays_path=overlays_dir.name,
        notes=notes,
    )
    manifest.save(manifest_path)

    return RunPaths(
        root=run_root,
        manifest=manifest_path,
        metrics=metrics_path,
        overlays=overlays_dir,
        config=config_path,
    )
