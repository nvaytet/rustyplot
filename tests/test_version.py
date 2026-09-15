"""The package version comes from the git tag, via hatch-vcs.

`__init__` reads it back with `importlib.metadata`, which only works for an
installed distribution; running from a plain checkout falls back to a
placeholder. Both paths are worth pinning down, since a broken `__version__`
would otherwise only show up after a release.
"""

import rustyplot


def test_version_is_a_non_empty_string():
    assert isinstance(rustyplot.__version__, str)
    assert rustyplot.__version__


def test_version_is_not_hardcoded_in_the_source():
    """Guards the hatch-vcs setup: a literal here would silently go stale."""
    source = (
        __import__("pathlib")
        .Path(rustyplot.__file__)
        .read_text()
    )
    assert '__version__ = "0.1.0"' not in source
    assert "_metadata.version" in source
