"""The two blind spots Batch 6/7 found in the phii-identity trajectory comparator.

Both are cases where the reported number was correct but MEANT something other
than what a reader would take it for:

  * SDHS changed its phase SEQUENCE under C3c (stock handed over to
    transition/momentum; C3c stayed energy-driven) while both arms kept the
    fate ``stopping_time``. A fate-only check reported no difference.
  * PRB reported dR2_max = 5661% while both arms were collapsing to the SAME
    0.01 pc floor. Once an arm is pinned there, a ratio against it measures the
    floor, not a divergence.

These tests build minimal synthetic runs rather than fixtures, so they stay fast
and do not depend on any committed output tree.
"""

import importlib.util
import json
from pathlib import Path

import pytest

HARNESS = (Path(__file__).resolve().parents[1]
           / "docs/dev/phii-identity/harness/compare_trajectories.py")

# docs/dev is untracked (local-only, see .gitignore) as of `a32b098`: absent in a
# fresh clone and in CI. Every test here drives that harness, so skip the module.
if not HARNESS.is_file():
    pytest.skip(
        "docs/dev is untracked (local-only); compare_trajectories.py unavailable",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def cmp_mod():
    spec = importlib.util.spec_from_file_location("compare_trajectories", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_run(root, name, rows, fate):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "dictionary.jsonl").open("w") as fh:
        for t, r2, phase in rows:
            fh.write(json.dumps({"t_now": t, "R2": r2, "current_phase": phase}) + "\n")
    (d / "metadata.json").write_text(json.dumps({"termination": {"outcome": fate}}))
    return d


def _growing(phases, n=40, t0=1e-3, t1=1.5):
    """A monotonically expanding shell, phase labels spread evenly over the run."""
    rows = []
    for i in range(n):
        f = i / (n - 1)
        t = t0 * (t1 / t0) ** f
        rows.append((t, 0.5 + 20.0 * f, phases[min(int(f * len(phases)), len(phases) - 1)]))
    return rows


def test_phase_sequence_change_is_caught_even_when_fate_matches(tmp_path, cmp_mod):
    """The SDHS case: same fate, different route. Must not read as no-difference."""
    base = _write_run(tmp_path / "base", "SDHS",
                      _growing(["energy", "implicit", "transition", "momentum"]),
                      "stopping_time")
    new = _write_run(tmp_path / "new", "SDHS",
                     _growing(["energy", "implicit"]), "stopping_time")

    r = cmp_mod.compare(base, new)

    assert r["fate_base"] == r["fate_new"], "precondition: the fate must NOT differ"
    assert r["verdict"] == "PHASE-CHANGE"
    assert r["phases_base"] == "energy>implicit>transition>momentum"
    assert r["phases_new"] == "energy>implicit"
    assert "phase sequence differs" in r["note"]


def test_identical_phase_sequence_does_not_trip_phase_change(tmp_path, cmp_mod):
    seq = ["energy", "implicit", "transition"]
    base = _write_run(tmp_path / "base", "X", _growing(seq), "stopping_time")
    new = _write_run(tmp_path / "new", "X", _growing(seq), "stopping_time")

    r = cmp_mod.compare(base, new)

    assert r["verdict"] != "PHASE-CHANGE"
    assert r["phases_base"] == r["phases_new"]


def test_collapse_floor_is_flagged_when_an_arm_is_pinned(tmp_path, cmp_mod):
    """The PRB case: both arms reach the same floor, one later than the other.

    The percentage explodes against a constant. It must be labelled, not reported
    as a bare divergence.
    """
    floor = cmp_mod.FLOOR_PC

    def collapsing(t_collapse):
        rows, n = [], 40
        for i in range(n):
            f = i / (n - 1)
            t = 1e-3 * (1.5 / 1e-3) ** f
            r2 = 5.0 if t < t_collapse else floor
            rows.append((t, r2, "energy"))
        return rows

    base = _write_run(tmp_path / "base", "PRB", collapsing(0.05), "shell_collapsed")
    new = _write_run(tmp_path / "new", "PRB", collapsing(0.8), "shell_collapsed")

    r = cmp_mod.compare(base, new)

    assert float(r["floor_grid_pct"]) > 0.0
    assert "COLLAPSE-FLOOR ARTIFACT" in r["note"]


def test_growing_run_is_not_mistaken_for_a_collapse_floor(tmp_path, cmp_mod):
    """Every run STARTS below the floor. Radius alone is not the discriminator.

    B3M grows monotonically to ~23 pc and never collapses, yet scored 19.2%
    "on floor" before the peak-time condition was added.
    """
    seq = ["energy", "implicit", "transition", "momentum"]
    rows = [(t, r2, ph) for t, r2, ph in _growing(seq)]
    rows[0] = (rows[0][0], cmp_mod.FLOOR_PC / 2, rows[0][2])  # starts below the floor

    base = _write_run(tmp_path / "base", "B3M", rows, "stopping_time")
    new = _write_run(tmp_path / "new", "B3M", rows, "stopping_time")

    r = cmp_mod.compare(base, new)

    assert float(r["floor_grid_pct"]) == 0.0
    assert "COLLAPSE-FLOOR" not in r["note"]
