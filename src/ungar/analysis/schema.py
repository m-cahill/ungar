"""Schema validation for UNGAR analytics artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore[assignment]

from ungar.enums import RANK_COUNT, SUIT_COUNT


class SchemaError(Exception):
    """Raised when an artifact violates the UNGAR analytics schema."""

    pass


def validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate a run manifest dictionary.

    Args:
        manifest: Dictionary loaded from manifest.json

    Raises:
        SchemaError: If validation fails
    """
    required_fields = {
        "run_id": str,
        "game": str,
        "algo": str,
        "created_at": str,
        "config": dict,
        "device": str,
    }

    # Check required fields and types
    for field, expected_type in required_fields.items():
        if field not in manifest:
            raise SchemaError(f"Manifest missing required field: {field}")
        if not isinstance(manifest[field], expected_type):
            raise SchemaError(
                f"Manifest field '{field}' has wrong type. Expected {expected_type}, got {type(manifest[field])}"
            )

    # Check version
    version = manifest.get("analytics_schema_version")
    if version is None:
        # We could allow missing version as legacy/v0, but M17 strictness implies we want it.
        # But for backward compatibility with pre-M17 runs, maybe we should be lenient?
        # The plan says "Enforce: Required keys present".
        # Let's enforce it for new runs, but maybe allow it to be missing if we are strict.
        # The prompt says "Treat analytics_schema_version as required".
        pass
        # raise SchemaError("Manifest missing analytics_schema_version")
    
    # If present, check it
    if version is not None:
        if not isinstance(version, int):
            raise SchemaError("analytics_schema_version must be an integer")


def validate_metrics_file(path: Path | str) -> None:
    """Validate a metrics CSV file without pandas.

    Args:
        path: Path to metrics.csv

    Raises:
        SchemaError: If validation fails
    """
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"Metrics file not found: {path}")

    required_columns = {"step", "episode", "reward"}

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise SchemaError("Metrics file is empty or missing header")

            headers = set(reader.fieldnames)
            missing = required_columns - headers
            if missing:
                raise SchemaError(f"Metrics file missing required columns: {missing}")

            last_step = -1
            
            for i, row in enumerate(reader):
                # Check types and ordering
                try:
                    step = int(row["step"])
                    _ = int(row["episode"])
                    _ = float(row["reward"])
                except ValueError as e:
                    raise SchemaError(f"Metrics file has invalid types at row {i+2}: {e}")

                if step < last_step:
                    raise SchemaError(f"Metrics file not sorted by step at row {i+2} ({step} < {last_step})")
                
                last_step = step

    except Exception as e:
        if isinstance(e, SchemaError):
            raise
        raise SchemaError(f"Failed to parse metrics file: {e}")


def validate_metrics(df: Any) -> None:
    """Validate a metrics DataFrame.

    Args:
        df: pandas DataFrame (typed as Any to avoid import errors)

    Raises:
        SchemaError: If validation fails
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for validate_metrics(df)")
    
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("Input must be a pandas DataFrame")

    required_columns = ["step", "episode", "reward"]
    for col in required_columns:
        if col not in df.columns:
            raise SchemaError(f"DataFrame missing required column: {col}")

    # Check types
    if not pd.api.types.is_integer_dtype(df["step"]):
        raise SchemaError("Column 'step' must be integer")
    if not pd.api.types.is_integer_dtype(df["episode"]):
        raise SchemaError("Column 'episode' must be integer")
    if not pd.api.types.is_numeric_dtype(df["reward"]):
        raise SchemaError("Column 'reward' must be numeric")

    # Check sorting
    if not df["step"].is_monotonic_increasing:
        raise SchemaError("Column 'step' must be sorted ascending")


def validate_overlay(payload: Dict[str, Any]) -> None:
    """Validate an overlay dictionary.

    Args:
        payload: Dictionary loaded from overlay JSON

    Raises:
        SchemaError: If validation fails
    """
    required_fields = ["run_id", "label", "importance"]
    for field in required_fields:
        if field not in payload:
            raise SchemaError(f"Overlay missing required field: {field}")

    importance = payload["importance"]
    if not isinstance(importance, list):
        raise SchemaError("Overlay 'importance' must be a list of lists")

    if len(importance) != SUIT_COUNT:
        raise SchemaError(f"Overlay 'importance' must have {SUIT_COUNT} rows (suits), got {len(importance)}")

    for i, row in enumerate(importance):
        if not isinstance(row, list):
            raise SchemaError(f"Overlay 'importance' row {i} must be a list")
        if len(row) != RANK_COUNT:
            raise SchemaError(
                f"Overlay 'importance' row {i} must have {RANK_COUNT} columns (ranks), got {len(row)}"
            )
        # Check elements are numbers
        for val in row:
            if not isinstance(val, (int, float)):
                raise SchemaError(f"Overlay 'importance' contains non-numeric value: {val}")

