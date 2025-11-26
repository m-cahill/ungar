from ungar.version import get_version


def test_get_version_returns_non_empty_string() -> None:
    assert isinstance(get_version(), str)
    assert get_version() != ""

