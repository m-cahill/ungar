"""Version information for the UNGAR package."""

__all__ = ["__version__", "get_version"]

__version__ = "0.1.0"


def get_version() -> str:
    """Return the UNGAR package version string."""
    return __version__
