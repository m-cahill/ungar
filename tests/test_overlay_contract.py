"""Overlay contract tests (M23).

These tests ensure overlay structure and labels remain stable in v1.x.
"""

import json
from pathlib import Path

import numpy as np
from ungar.xai import overlay_from_dict

# ============================================================================
# Overlay Shape Contract
# ============================================================================


def test_overlay_importance_shape_is_4x14() -> None:
    """Test that overlay importance shape is exactly (4, 14)."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_overlay_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        overlay_dict = json.load(f)

    overlay = overlay_from_dict(overlay_dict)

    # v1 contract: shape MUST be (4, 14)
    assert overlay.importance.shape == (4, 14)


def test_overlay_importance_is_2d_array() -> None:
    """Test that importance is a 2D numpy array."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_overlay_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        overlay_dict = json.load(f)

    overlay = overlay_from_dict(overlay_dict)

    assert isinstance(overlay.importance, np.ndarray)
    assert overlay.importance.ndim == 2


# ============================================================================
# Overlay Label Contract
# ============================================================================


def test_standard_overlay_labels_defined() -> None:
    """Test that standard v1 overlay labels are defined."""
    from ungar.training.config import XAIConfig

    # v1 contract: these labels are standard
    standard_labels = ["heuristic", "random", "policy_grad", "value_grad"]

    for label in standard_labels:
        assert label in XAIConfig.VALID_METHODS


def test_overlay_label_is_string() -> None:
    """Test that overlay label is a string."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_overlay_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        overlay_dict = json.load(f)

    overlay = overlay_from_dict(overlay_dict)

    assert isinstance(overlay.label, str)
    assert len(overlay.label) > 0


# ============================================================================
# Overlay Required Fields Contract
# ============================================================================


def test_overlay_required_fields_present() -> None:
    """Test that all required overlay fields exist."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_overlay_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        overlay_dict = json.load(f)

    # v1 contract: these fields are REQUIRED
    required_fields = ["run_id", "label", "agg", "step", "importance", "meta"]

    for field in required_fields:
        assert field in overlay_dict, f"Required field '{field}' missing from overlay"


def test_overlay_roundtrip_json() -> None:
    """Test that overlays can round-trip through JSON."""
    from ungar.xai import overlay_to_dict

    fixture_path = Path(__file__).parent / "fixtures" / "sample_overlay_v1.json"

    with open(fixture_path, "r", encoding="utf-8") as f:
        original_dict = json.load(f)

    # Load as overlay
    overlay = overlay_from_dict(original_dict)

    # Convert back to dict
    roundtrip_dict = overlay_to_dict(overlay)

    # Key fields should match
    assert roundtrip_dict["run_id"] == original_dict["run_id"]
    assert roundtrip_dict["label"] == original_dict["label"]
    assert roundtrip_dict["step"] == original_dict["step"]
    # Importance in dict form is a list, check length
    assert len(roundtrip_dict["importance"]) == 4  # 4 suits
    assert len(roundtrip_dict["importance"][0]) == 14  # 14 ranks
