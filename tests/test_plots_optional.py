"""Tests for optional plotting dependencies."""

import importlib
import sys
from typing import Generator

import pytest


@pytest.fixture
def mock_no_matplotlib() -> Generator[None, None, None]:
    """Simulate an environment without matplotlib."""
    with pytest.MonkeyPatch.context() as m:
        m.setitem(sys.modules, "matplotlib", None)
        m.setitem(sys.modules, "matplotlib.pyplot", None)
        yield


def test_plot_module_imports_without_matplotlib(mock_no_matplotlib: None) -> None:
    """Module should import even if matplotlib is missing."""
    # We must reload because it might be already imported
    if "ungar.analysis.plots" in sys.modules:
        importlib.reload(sys.modules["ungar.analysis.plots"])
    else:
        pass

    import ungar.analysis.plots as plots

    assert hasattr(plots, "_require_matplotlib")
    assert plots.plt is None


def test_plot_functions_complain_without_matplotlib(mock_no_matplotlib: None) -> None:
    """Plotting functions should raise RuntimeError if matplotlib is missing."""
    if "ungar.analysis.plots" in sys.modules:
        importlib.reload(sys.modules["ungar.analysis.plots"])

    import ungar.analysis.plots as plots

    # Reload to ensure plt is None
    importlib.reload(plots)
    assert plots.plt is None

    with pytest.raises(RuntimeError, match="Matplotlib is required"):
        plots.plot_learning_curve([], [])

    # Restore original module
    importlib.reload(plots)
