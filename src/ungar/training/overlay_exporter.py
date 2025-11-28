"""Backend XAI Overlay Exporter.

Handles generation and serialization of CardOverlay artifacts during training,
independent of any frontend dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Protocol

from ungar.xai import CardOverlay, overlay_to_dict


class OverlayGenerator(Protocol):
    """Protocol for generating overlays from game states."""

    def generate(self, state: Any) -> CardOverlay:
        ...


class OverlayExporter:
    """Manages collection and export of XAI overlays."""

    def __init__(self, export_dir: str | Path) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.overlays: List[CardOverlay] = []

    def add(self, overlay: CardOverlay) -> None:
        """Add an overlay to the collection."""
        self.overlays.append(overlay)

    def save(self, filename: str = "overlays.json") -> Path:
        """Serialize collected overlays to JSON file."""
        output_path = self.export_dir / filename
        data = [overlay_to_dict(o) for o in self.overlays]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return output_path

    def clear(self) -> None:
        """Clear collected overlays."""
        self.overlays.clear()
