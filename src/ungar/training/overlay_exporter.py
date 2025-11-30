"""Backend XAI Overlay Exporter.

Handles generation and serialization of CardOverlay artifacts during training,
independent of any frontend dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import numpy as np
from ungar.xai import overlay_to_dict
from ungar.xai_methods import OverlayMethod


class OverlayExporter:
    """Manages collection and export of XAI overlays."""

    def __init__(
        self,
        out_dir: str | Path,
        methods: List[OverlayMethod] | None = None,
        max_overlays: int = 200,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.methods = methods or []
        self.max_overlays = max_overlays
        self.count = 0

    def export(
        self,
        obs: np.ndarray,
        action: int,
        step: int,
        run_id: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Generate and save overlays using configured methods."""
        if self.count >= self.max_overlays:
            return

        for method in self.methods:
            if self.count >= self.max_overlays:
                break

            overlay = method.compute(
                obs=obs,
                action=action,
                step=step,
                run_id=run_id,
                meta=meta,
            )
            
            # Save individual file per overlay as per plan M19-B
            # runs/<run_id>/overlays/<label>_<step>.json
            # But we might have multiple methods, so include label in filename
            filename = f"{method.label}_{step}.json"
            output_path = self.out_dir / filename
            
            data = overlay_to_dict(overlay)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            self.count += 1
