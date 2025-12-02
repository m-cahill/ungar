"""Analytics schema v1 contract tests (M23).

These tests ensure the analytics schema v1 contract remains stable.
Breaking these tests indicates a breaking change requiring version bump.
"""

import json
from pathlib import Path

# ============================================================================
# Schema Version Contract
# ============================================================================


def test_schema_version_is_one() -> None:
    """Test that schema version is exactly 1 for v1.x."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_manifest_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # v1 contract: schema version MUST be 1
    assert manifest["analytics_schema_version"] == 1


def test_manifest_required_fields_present() -> None:
    """Test that all required manifest fields exist."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_manifest_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # v1 contract: these fields are REQUIRED
    required_fields = [
        "analytics_schema_version",
        "run_id",
        "timestamp",
        "created_at",
        "game",
        "algo",
        "device",
        "config",
    ]

    for field in required_fields:
        assert field in manifest, f"Required field '{field}' missing from manifest"


def test_manifest_schema_version_is_int() -> None:
    """Test that schema_version is an integer."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_manifest_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert isinstance(manifest["analytics_schema_version"], int)


def test_manifest_validates_via_schema_module() -> None:
    """Test that manifest validates using UNGAR's schema validator."""
    from ungar.analysis.schema import validate_manifest

    fixture_path = Path(__file__).parent / "fixtures" / "sample_manifest_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Should not raise
    validate_manifest(manifest)
