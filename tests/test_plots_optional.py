"""Tests for optional plotting dependencies."""


def test_plot_module_imports_successfully() -> None:
    """Module should import even if matplotlib might be missing in CI."""
    # Simply importing the module should not fail even if matplotlib is missing
    # because we wrapped it in try/except.
    import ungar.analysis.plots as plots

    assert hasattr(plots, "_require_matplotlib")
    assert hasattr(plots, "plot_learning_curve")
    assert hasattr(plots, "plot_overlay_heatmap")
