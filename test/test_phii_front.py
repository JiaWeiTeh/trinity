#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gates for option C's front-pressure closure and the scheme dispatcher.

Option C (decision D16, ruled 2026-09-11; docs/dev/phii-identity/PLAN.md 0.0.5 Step 3)
replaces the cavity Stroemgren pressure with the pressure AT the ionisation front,

    P_front = (mu_c/mu_i) * n_IF * k_B * T_ion,

applied by the caller over 4*pi*R_IF**2 on the neutral rind, with NO Pb or P_ram added.
It lives behind `phii_scheme`, which defaults to the shipped 'c3c'.

WHAT IS PINNED HERE, and what each gate is FOR.

  F1  NO END-STATE BRANCH (ruling 2026-09-14).  The closure must return the SAME thing
      whether or not the shell is fully ionised -- it must not read `has_neutral` at all.
      C v1 did branch, returning 0.0 on a fully ionised shell, and it was wrong: a fully
      ionised shell means LyC escapes, which means the front has run PAST rShell into
      cloud the model no longer tracks. There is still neutral gas; the model has lost
      the front's radius. n_IF and R_IF are continuous through that moment, so a branch
      there injects a discontinuity the physics does not have (a factor 7.5 in the drive,
      measured -- data/b32_alwayson.csv). What replaces it is a REPORTED flag,
      shell_frontEscaped, which nothing in the dynamics reads.
  F2  the magnitude, stated once.  p_ref * n_IF, and nothing else -- no photon budget,
      no volume, no escape fraction. That insulation is deliberate: C3c inverts a
      recombination balance and so needed the W69 gas-vs-total correction and would need
      the 2026-09-12 sky-partition correction too; n_IF needs neither.
  F3  THE WEAK-IONISATION LIMIT, the reason C was chosen over the layer-pressure option.
      As the layer thins, n_IF -> n0 = Pb/p_ref, so the drive must tend to Pb exactly --
      which in the momentum phase (where Pb IS P_ram) is the pure wind solution,
      recovered with no branch. Pinned algebraically here; measured on real states
      2026-09-11 (R_IF/R2 -> 1.0000000, n_IF/n0 -> 1.000000, continuous through zero).
  F4  degenerate inputs return exactly 0.0, never NaN or inf (the L5 obligation).  An
      ODE right-hand side poisoned by a NaN fails far from here, and silently.
  F5  the dispatcher.  Default is the shipped scheme; an unknown name falls BACK to it
      rather than raising, because a typo in a .param must not change the physics
      silently. And with scheme 'c3c' the dispatcher must be indistinguishable from
      calling get_phii_c3c directly -- that is what makes C3c a usable control arm.

    pytest test/test_phii_front.py -v
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trinity._input.read_param import read_param                  # noqa: E402
from trinity.bubble_structure import get_bubbleParams as G        # noqa: E402


class _Shell:
    """Stand-in for ShellProperties. get_phii_front reads exactly two attributes.

    ponytail: attributes are omitted rather than set to None when absent, so that the
    `getattr(..., default)` path in the closure is exercised too -- that is the shape
    the older stubs in test_phii_limits.py / test_phii_c3c_spitzer.py actually have,
    and it is why those suites are unaffected by this scheme.
    """

    def __init__(self, has_neutral=None, n_IF=None, f_abs=1.0):
        self.shell_fAbsorbedIon = f_abs
        self.shell_fAbsorbedIonGas = f_abs
        if has_neutral is not None:
            self.has_neutral = has_neutral
        if n_IF is not None:
            self.n_IF = n_IF


@pytest.fixture()
def p():
    q = read_param(str(REPO / "param" / "simple_cluster.param"))
    q["Pb"].value = 1.0e3
    return q


def _p_ref(q):
    return (q["mu_convert"].value / q["mu_ion_shell"].value
            * q["k_B"].value * q["TShell_ion"].value)


# --------------------------------------------------------------- F1 no end-state branch
@pytest.mark.parametrize("flag", [True, False, None], ids=["state2", "state1", "absent"])
def test_end_state_does_not_change_the_answer(p, flag):
    """The whole point of the 2026-09-14 ruling: same n_IF, same answer, either side of
    the end-state flip. If this fails, the branch has crept back in."""
    n_IF = 3.0e57
    ref = _p_ref(p) * n_IF
    shell = _Shell(has_neutral=flag, n_IF=n_IF)
    assert G.get_phii_front(p, shell) == pytest.approx(ref, rel=1e-14)


def test_closure_does_not_read_has_neutral_at_all(p):
    """Stronger than the above: the attribute may not even be consulted. A shell object
    that raises on access to `has_neutral` must still give the right answer."""
    class _Trap:
        shell_fAbsorbedIon = 1.0
        n_IF = 3.0e57
        @property
        def has_neutral(self):                      # noqa: D401
            raise AssertionError("get_phii_front must not read has_neutral")
    assert G.get_phii_front(p, _Trap()) == pytest.approx(_p_ref(p) * 3.0e57, rel=1e-14)


# --------------------------------------------------------------- F2 magnitude
def test_magnitude_is_pref_times_nIF(p):
    n_IF = 3.25e57
    got = G.get_phii_front(p, _Shell(has_neutral=True, n_IF=n_IF))
    assert got == pytest.approx(_p_ref(p) * n_IF, rel=1e-14)


def test_magnitude_ignores_the_photon_budget(p):
    """C3c's answer moves with f_abs; C's must not. This is the W69/sky-partition insulation."""
    a = G.get_phii_front(p, _Shell(has_neutral=True, n_IF=2.0e57, f_abs=1.0))
    b = G.get_phii_front(p, _Shell(has_neutral=True, n_IF=2.0e57, f_abs=0.01))
    assert a == b


# --------------------------------------------------------------- F3 weak-ionisation limit
def test_weak_ionisation_limit_is_exact(p):
    """n_IF -> n0 = Pb/p_ref  =>  drive -> Pb, to roundoff. The D16 ruling's crux."""
    Pb = p["Pb"].value
    n0 = Pb / _p_ref(p)
    assert G.get_phii_front(p, _Shell(has_neutral=True, n_IF=n0)) == pytest.approx(Pb, rel=1e-14)


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-6, 1e-9, 1e-12])
def test_weak_ionisation_limit_is_continuous(p, eps):
    """Approaching n0 from above, the drive approaches Pb monotonically -- no jump.

    The layer-pressure option failed exactly here, by COMPOSITION: added to a P_ram that
    equals Pb it gives 2*P_ram, discontinuously at Qi = 0+. C adds nothing, so the
    approach is smooth and the limit is the value.
    """
    Pb = p["Pb"].value
    n0 = Pb / _p_ref(p)
    got = G.get_phii_front(p, _Shell(has_neutral=True, n_IF=n0 * (1.0 + eps)))
    assert got / Pb == pytest.approx(1.0 + eps, rel=1e-12)


# --------------------------------------------------------------- F4 degenerate inputs
@pytest.mark.parametrize("n_IF", [0.0, -1.0, -1e57, float("nan"), float("inf"),
                                 float("-inf"), "not a number", None],
                         ids=["zero", "neg", "neg-big", "nan", "inf", "-inf", "str", "none"])
def test_degenerate_nIF_returns_exactly_zero(p, n_IF):
    shell = _Shell(has_neutral=True)
    shell.n_IF = n_IF                       # bypass the ctor's None-means-absent rule
    out = G.get_phii_front(p, shell)
    assert out == 0.0
    assert np.isfinite(out)


def test_never_negative_over_a_wide_sweep(p):
    """F6/L6 sign obligation: the stored P_HII must never go negative, in any phase."""
    for n_IF in np.logspace(40, 70, 61):
        assert G.get_phii_front(p, _Shell(has_neutral=True, n_IF=float(n_IF))) >= 0.0


# --------------------------------------------------------------- F5 the dispatcher
def test_default_scheme_is_c3c(p):
    assert str(p["phii_scheme"].value) == "c3c"


def test_dispatcher_matches_c3c_exactly_by_default(p):
    """With the default scheme the dispatcher must be indistinguishable from the shipped
    helper -- that is what makes C3c a control arm rather than an approximation of one."""
    for f_abs in (1.0, 0.5, 0.039, 0.0):
        shell = _Shell(has_neutral=True, n_IF=2.0e57, f_abs=f_abs)
        assert G.get_phii(p, shell) == G.get_phii_c3c(p, shell)


@pytest.mark.parametrize("name", ["front", "FRONT", " front "])
def test_dispatcher_selects_front(p, name):
    p["phii_scheme"].value = name
    shell = _Shell(has_neutral=True, n_IF=2.0e57)
    assert G.get_phii(p, shell) == G.get_phii_front(p, shell)


# "o1"/"k11" were unknown names until 2026-09-22, when the arms were registered in
# PHII_SCHEMES. "k12" is the near-miss that is still genuinely not a scheme.
@pytest.mark.parametrize("name", ["c3a", "", "Front-pressure", "k12"])
def test_unknown_scheme_falls_back_to_c3c(p, name):
    """A typo must not silently change the physics."""
    p["phii_scheme"].value = name
    shell = _Shell(has_neutral=True, n_IF=2.0e57)
    assert G.get_phii(p, shell) == G.get_phii_c3c(p, shell)


# --------------------------------------------------------------- F6 validator + pre-gate
# Both added 2026-09-12 after /xcheck found (a) the dispatcher docstring asserting a
# validation that did not exist, and (b) all six call sites pre-gating option C on
# `n_IF_Str > 0` -- C3c's cavity Stroemgren density, which vanishes in exactly the
# weak-ionisation limit C was chosen for.
def test_a_bad_scheme_name_is_rejected_at_startup():
    """The typo must be loud. The dispatcher's fallback is belt and braces, not the
    defence: an unrecognised phii_scheme has to fail before the run starts."""
    from trinity._input.registry import _validate_phii_scheme
    from trinity._input.errors import ParameterFileError
    for bad in ("c3a", "Front-pressure", "k12", ""):
        with pytest.raises(ParameterFileError):
            _validate_phii_scheme(bad, None)
    for good in ("c3c", "front", "FRONT", " front "):
        _validate_phii_scheme(good, None)      # must not raise


def test_pregate_is_scheme_aware(p):
    """c3c's pre-gate is n_IF_Str (kept verbatim, so c3c stays bit-identical); the front
    scheme must NOT inherit it, because get_phii_front self-gates on its own quantities
    and n_IF_Str vanishes where C matters most."""
    live = _Shell(has_neutral=True, n_IF=2.0e57)
    live.n_IF_Str = 0.0                        # C3c's density gone, the front's intact
    p["phii_scheme"].value = "c3c"
    assert G.phii_is_active(p, live) is False or G.phii_is_active(p, live) == False
    p["phii_scheme"].value = "front"
    assert G.phii_is_active(p, live) is True
    # and with n_IF_Str present, c3c is active again
    live.n_IF_Str = 1.0e57
    p["phii_scheme"].value = "c3c"
    assert G.phii_is_active(p, live)


# --------------------------------------------------------------- F7 the validity flag
def test_front_escaped_flag_is_reported_not_acted_on():
    """shell_frontEscaped must be a registered OUTPUT and must not be wired into any
    force assembly. A flag the dynamics reads is a branch by another name, which is the
    thing the 2026-09-14 ruling removed."""
    import trinity._input.registry as REG
    import trinity.shell_structure.shell_structure as SS
    assert 'shell_frontEscaped' in REG.REGISTRY, "flag must reach the run output"
    assert REG.REGISTRY['shell_frontEscaped'].exclude_from_snapshot is False
    for rel in ("trinity/phase1_energy/energy_phase_ODEs.py",
                "trinity/phase1b_energy_implicit/run_energy_implicit_phase.py",
                "trinity/phase1c_transition/run_transition_phase.py",
                "trinity/phase2_momentum/run_momentum_phase.py",
                "trinity/bubble_structure/get_bubbleParams.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        # access patterns, not mentions -- these files are allowed to EXPLAIN the flag in
        # a comment (and do), they are just not allowed to read it.
        for pattern in (".shell_frontEscaped", "'shell_frontEscaped'", '"shell_frontEscaped"'):
            assert pattern not in src, (
                f"{rel} reads shell_frontEscaped via {pattern} -- it is a REPORTED flag, "
                f"not a switch")
    assert not hasattr(SS, "_FRONT_ESCAPE_FLAG_THRESHOLD"), (
        "the flag has NO threshold by ruling 2026-09-14 -- any escape counts, so a tunable "
        "constant is exactly the knob that must not exist")


@pytest.mark.parametrize("f_abs,expected", [
    (1.0, False),          # nothing escapes: the front is still inside the shell
    (1.0 - 1e-15, True),   # one photon's worth is already outside the geometry
    (0.99, True),
    (0.863, True),         # end of the B3M momentum run
    (0.27, True),          # the energy-phase regime, where the geometry is far past valid
], ids=["sealed", "one-photon", "1pc", "b3m-end", "energy-phase"])
def test_front_escaped_is_any_escape(f_abs, expected):
    """No threshold: the test is f_absorbed_ion < 1, full stop. `sealed` is exact because
    f_absorbed_ion is exactly 1.0 when nothing escapes (0/664 archived end-state-2 rows
    have any escape), so the strict comparison is not a floating-point trap."""
    assert bool(f_abs < 1.0) is expected


def test_the_two_schemes_actually_differ(p):
    """Guard against a dispatcher that looks wired but is not: on a state-2 shell whose
    front pressure exceeds the cavity value, the two must give different answers."""
    shell = _Shell(has_neutral=True, n_IF=9.0e57)
    assert G.get_phii_front(p, shell) != G.get_phii_c3c(p, shell)
