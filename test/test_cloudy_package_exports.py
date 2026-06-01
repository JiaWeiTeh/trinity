"""
Tests that the ``trinity._output.cloudy`` package re-exports the public API
so users can do ``from trinity._output.cloudy import build_dlaw_block`` etc.

Trivial smoke check, but guards against accidental dropping from __all__
when refactoring.
"""

import trinity._output.cloudy as cloudy


def test_package_reexports_public_api():
    expected = {
        "build_dlaw_block", "DlawError",
        "load_run", "RunBundle", "RunLoadError",
        "snapshot_to_values", "SnapshotInvalid",
    }
    assert expected.issubset(set(dir(cloudy)))


def test_package_all_matches_exports():
    """Everything in __all__ should be importable from the package root."""
    for name in cloudy.__all__:
        assert hasattr(cloudy, name), f"package missing {name!r} from __all__"


def test_package_re_exports_are_the_same_objects():
    """Re-exports must point at the canonical objects (not shadowed copies)."""
    from trinity._output.cloudy.dlaw import build_dlaw_block, DlawError
    from trinity._output.cloudy.run_loader import (
        RunBundle, RunLoadError, load_run,
    )
    from trinity._output.cloudy.snapshot_to_deck import (
        SnapshotInvalid, snapshot_to_values,
    )
    assert cloudy.build_dlaw_block is build_dlaw_block
    assert cloudy.DlawError is DlawError
    assert cloudy.load_run is load_run
    assert cloudy.RunBundle is RunBundle
    assert cloudy.RunLoadError is RunLoadError
    assert cloudy.snapshot_to_values is snapshot_to_values
    assert cloudy.SnapshotInvalid is SnapshotInvalid
