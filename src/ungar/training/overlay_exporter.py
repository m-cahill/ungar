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
    """Manages collection and export of XAI overlays (M22: with optional batching)."""

    def __init__(
        self,
        out_dir: str | Path,
        methods: List[OverlayMethod] | None = None,
        max_overlays: int = 200,
        batch_size: int | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.methods = methods or []
        self.max_overlays = max_overlays
        self.batch_size = batch_size
        self.count = 0

        # M22: Per-method buffers for batching
        self.buffers: dict[str, list[dict[str, Any]]] = {
            method.label: [] for method in self.methods
        }

    def export(
        self,
        obs: np.ndarray,
        action: int,
        step: int,
        run_id: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Generate and save overlays using configured methods (M22: with optional batching)."""
        if self.count >= self.max_overlays:
            return

        for method in self.methods:
            if self.count >= self.max_overlays:
                break

            # M22: If batching is disabled (batch_size=None), use sequential path
            if self.batch_size is None:
                overlay = method.compute(
                    obs=obs,
                    action=action,
                    step=step,
                    run_id=run_id,
                    meta=meta,
                )
                self._save_overlay(overlay, method.label, step)
                self.count += 1
            else:
                # M22: Add to buffer and flush if full
                buffer_item = {
                    "obs": obs.copy(),  # Copy to avoid mutation
                    "action": action,
                    "step": step,
                    "run_id": run_id,
                    "meta": meta,
                }
                self.buffers[method.label].append(buffer_item)

                # Flush buffer if it reaches batch_size
                if len(self.buffers[method.label]) >= self.batch_size:
                    self._flush_buffer(method)

    def _save_overlay(self, overlay: Any, label: str, step: int) -> None:
        """Save a single overlay to disk."""
        filename = f"{label}_{step}.json"
        output_path = self.out_dir / filename

        data = overlay_to_dict(overlay)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _flush_buffer(self, method: OverlayMethod) -> None:
        """Flush buffered overlay requests for a specific method (M22)."""
        buffer = self.buffers[method.label]
        if not buffer:
            return

        # Compute batch of overlays
        overlays = method.compute_batch(buffer)

        # Save each overlay
        for overlay in overlays:
            if self.count >= self.max_overlays:
                break
            self._save_overlay(overlay, method.label, overlay.step)
            self.count += 1

        # Clear buffer
        self.buffers[method.label] = []

    def flush(self) -> None:
        """Flush all partial batches at end of training (M22)."""
        for method in self.methods:
            if self.buffers[method.label]:
                self._flush_buffer(method)
