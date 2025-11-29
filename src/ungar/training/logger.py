"""Training logger abstraction.

Supports logging metrics to CSV, JSON lines, and TensorBoard (if available).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Protocol

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore


class TrainingLogger(Protocol):
    """Protocol for logging training metrics."""

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log a dictionary of scalar metrics."""
        ...

    def close(self) -> None:
        """Close logger and flush data."""
        ...


class NoOpLogger:
    """Dummy logger that does nothing."""

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class FileLogger:
    """Logs metrics to a CSV or JSONL file."""

    def __init__(
        self, log_dir: str | Path, format: str = "csv", filename: str | None = None
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.format = format

        if filename:
            self.log_file = self.log_dir / filename
        else:
            timestamp = int(time.time())
            self.log_file = self.log_dir / f"metrics_{timestamp}.{format}"

        self.file = open(self.log_file, "w", newline="", encoding="utf-8")
        self.writer: Any = None

        if format == "csv":
            # Will initialize writer on first log to know headers
            pass

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        row = {"step": step, **metrics}

        if self.format == "csv":
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
                self.writer.writeheader()
            self.writer.writerow(row)
        elif self.format == "jsonl":
            self.file.write(json.dumps(row) + "\n")

        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()


class TensorBoardLogger:
    """Logs metrics to TensorBoard."""

    def __init__(self, log_dir: str | Path) -> None:
        if not HAS_TENSORBOARD:
            raise ImportError(
                "tensorboard is not installed. Install with `pip install tensorboard`."
            )
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self) -> None:
        self.writer.close()


class MultiLogger:
    """Combines multiple loggers."""

    def __init__(self, loggers: list[TrainingLogger]) -> None:
        self.loggers = loggers

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
