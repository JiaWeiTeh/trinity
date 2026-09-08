"""isCollapse must be a STATE, not a one-way latch.

Before 2026-09-08 the three integrator loops only ever set isCollapse True. A
shell that dipped inward once early stayed flagged for the rest of the run:
mShell was frozen in the ODE right-hand sides from that point, and every
downstream summary reported a blow-out as a collapse (156 of grid E2's runs
crossed the read-off radius outward at 5-6 km/s and stopped at stop_r still
carrying isCollapse=True). Guard both halves of the transition here, and guard
that all three phases still carry the release branch.
"""
import pathlib
import re

PHASES = [
    "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
    "trinity/phase1c_transition/run_transition_phase.py",
    "trinity/phase2_momentum/run_momentum_phase.py",
]
REPO = pathlib.Path(__file__).resolve().parents[1]


def _step(flag, v2, R2, R2_prev):
    """The transition as written in all three loops."""
    if v2 < 0 and R2 < R2_prev:
        return True
    elif v2 > 0 and R2 > R2_prev:
        return False
    return flag


def test_latch_sets_and_releases():
    assert _step(False, -1.0, 0.10, 0.20) is True, "inward motion must set the flag"
    assert _step(True, 1.0, 0.30, 0.20) is False, "re-expansion must clear it"
    # ambiguous steps hold the previous state rather than flapping
    assert _step(True, 1.0, 0.10, 0.20) is True
    assert _step(False, -1.0, 0.30, 0.20) is False
    # the case that motivated the fix: dip once, then blow out and stay out
    flag = False
    for v2, R2, prev in [(-1, 0.02, 0.03), (2, 0.05, 0.02), (3, 0.28, 0.05)]:
        flag = _step(flag, v2, R2, prev)
    assert flag is False, "a shell that dipped once must not stay flagged forever"


def test_every_phase_still_releases():
    for rel in PHASES:
        src = (REPO / rel).read_text()
        assert re.search(r"elif v2 > 0 and R2 > R2_prev:\s*\n\s*params\['isCollapse'\]\.value = False", src), \
            f"{rel} lost the isCollapse release branch"


if __name__ == "__main__":
    test_latch_sets_and_releases()
    test_every_phase_still_releases()
    print("OK: isCollapse sets, releases, and all three phases carry the branch")
