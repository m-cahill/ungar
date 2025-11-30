"""Tests for analytics schema validation."""

import csv
import json
import pytest
from pathlib import Path

from ungar.analysis.schema import (
    SchemaError,
    validate_manifest,
    validate_metrics_file,
    validate_overlay,
)


@pytest.mark.smoke
def test_validate_manifest_valid():
    manifest = {
        "run_id": "test_run",
        "game": "high_card",
        "algo": "dqn",
        "created_at": "2025-01-01T00:00:00Z",
        "config": {"lr": 0.01},
        "device": "cpu",
        "analytics_schema_version": 1,
    }
    # Should not raise
    validate_manifest(manifest)


def test_validate_manifest_missing_field():
    manifest = {
        "run_id": "test_run",
        "game": "high_card",
        # Missing algo
        "created_at": "2025-01-01T00:00:00Z",
        "config": {},
        "device": "cpu",
    }
    with pytest.raises(SchemaError, match="missing required field"):
        validate_manifest(manifest)


def test_validate_manifest_wrong_type():
    manifest = {
        "run_id": 123,  # Should be string
        "game": "high_card",
        "algo": "dqn",
        "created_at": "2025-01-01T00:00:00Z",
        "config": {},
        "device": "cpu",
    }
    with pytest.raises(SchemaError, match="wrong type"):
        validate_manifest(manifest)


def test_validate_metrics_file_valid(tmp_path: Path):
    metrics_path = tmp_path / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "reward", "loss"])
        writer.writerow([1, 1, 0.5, 0.1])
        writer.writerow([2, 1, 0.6, 0.05])
        writer.writerow([3, 2, 0.7, 0.04])
    
    # Should not raise
    validate_metrics_file(metrics_path)


def test_validate_metrics_file_missing_header(tmp_path: Path):
    metrics_path = tmp_path / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        f.write("1,1,0.5\n")
    
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_metrics_file(metrics_path)


def test_validate_metrics_file_unsorted(tmp_path: Path):
    metrics_path = tmp_path / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode", "reward"])
        writer.writerow([2, 1, 0.6])
        writer.writerow([1, 1, 0.5])  # Unsorted
    
    with pytest.raises(SchemaError, match="not sorted"):
        validate_metrics_file(metrics_path)


def test_validate_overlay_valid():
    # 4 suits, 14 ranks
    importance = [[0.0] * 14 for _ in range(4)]
    overlay = {
        "run_id": "test",
        "label": "policy",
        "importance": importance,
    }
    # Should not raise
    validate_overlay(overlay)


def test_validate_overlay_bad_shape():
    # 3 suits instead of 4
    importance = [[0.0] * 14 for _ in range(3)]
    overlay = {
        "run_id": "test",
        "label": "policy",
        "importance": importance,
    }
    with pytest.raises(SchemaError, match="must have 4 rows"):
        validate_overlay(overlay)

    # 4 suits, but 13 ranks
    importance = [[0.0] * 13 for _ in range(4)]
    overlay = {
        "run_id": "test",
        "label": "policy",
        "importance": importance,
    }
    with pytest.raises(SchemaError, match="must have 14 columns"):
        validate_overlay(overlay)

