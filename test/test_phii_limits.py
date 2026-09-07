#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The limits SSOT for every P_HII scheme: shipped C3c and both arm candidates.

Why this file exists (docs/dev/phii-identity/PLAN.md section 0.0, Phase 2). The
workstream's limit evidence was FORKED: `test_phii_c3c_spitzer.py` pins the shipped
scheme, and a hand-copied variant under `docs/dev/.../harness/test_phii_k11_spitzer.py`
(differing only in its shell stub) pinned K11 -- so the sharpest publishable claim in
the workstream, "O1 loses the photoionisation-only limit BY CONSTRUCTION while K11
retains it exactly", had no single committed artifact behind it. Worse, the WIND-only
half of section 3's two-sided obligation had no unit test at all for any scheme; it
rested on a trajectory ladder's fitted exponent. This file is one suite over all three
schemes, so a limit is stated once and measured the same way for each.

    pytest test/test_phii_limits.py -v
    python test/test_phii_limits.py --out docs/dev/phii-identity/data/b23_limits.csv

WHAT IS PINNED, and what each gate is FOR

  L1  photoionisation-only.  Wind -> 0 (Lmech = 0, so P_conf = P_ram = 0). The drive
      must survive and equal the classical Stromgren pressure p_ref * n_St over the
      front. THIS IS THE STRUCTURAL RESULT: C3c and K11 pass, O1 CANNOT -- its drive is
      proportional to P_conf, so it follows the wind to zero. Recorded as an expected
      failure for O1, not as a bug to fix: no implementation choice repairs it.
  L2  wind-only.  Photons -> 0 (Qi -> 0, front collapses to R2). The COMPOSED driving
      pressure must CONVERGE to P_ram in the momentum phase. Tests the composition, not
      just the helper -- that is the half that had no test. Taken as a limit rather than
      at Qi = 0 exactly, because at exactly 0 all three schemes return through a shared
      guard and the gate passes without evaluating anything (found by audit, 2026-09-02).
  L3  strong-wind confined fork.  C_W >> C_Q with R_IF/R2 -> 1. RECORDED, NOT SCORED:
      the three schemes legitimately disagree here (C3c transmits and returns 0, O1
      returns P_conf*(R_IF/R2)^2, K11 returns P_ram with no area amplification) and
      choosing between them is D5, a physics-intent call the maintainer owns.
  L4  front collapse.  R_IF -> R2 must reduce K11's drive to the shipped P_C3a form and
      O1's to P_conf. This is what makes L1's dynamics transfer: the classical D-type
      integration (index 4/7, Hosokawa & Inutsuka to 1e-4, (4/3)^(2/7) above Spitzer)
      is already pinned for P_C3a in test_phii_c3c_spitzer.py, so a scheme that equals
      P_C3a to 1e-12 at every radius inherits it and need not re-integrate.
  L5  degenerate inputs return exactly 0.0, never NaN or inf. An ODE right-hand side
      poisoned by a NaN fails far from here, and silently.
  L6  sign and composition. The stored P_HII must never go negative, in momentum OR
      transition. W7 recorded the transition case as unbounded; re-auditing it, the
      bound exists and is provable for all three schemes (see the gate), so it is now
      asserted in both phases rather than merely measured. The margin is reported
      because it is what carries the proof.
  L7  continuity across the branch. NOT IMPLEMENTED HERE.
      ponytail: C3c's +6.79% state jump and both arms' continuity were already measured
      on committed trajectories (PLAN.md Batch 13 / Batch 21), and re-deriving them needs
      a trajectory, not a limit. Add it here only if a scheme's branch logic changes.

HOW THE ARMS ARE LOADED. Both candidates live ONLY as patches under
docs/dev/phii-identity/hpc/b14/, which is untracked since a32b098e. This file is in the
tracked test suite, so it must not depend on them: each arm's copy of
get_bubbleParams.py is built in a temp dir by `git apply`-ing the same patch
`run_arms.sh` applies, and the arm parametrisations SKIP when the patch is absent (a
fresh clone, or CI). The shipped scheme is always tested.
"""

import argparse
import csv
import os
import importlib.util
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from trinity._input.read_param import read_param  # noqa: E402

HELPER_REL = Path("trinity/bubble_structure/get_bubbleParams.py")
ARMS = REPO / "docs/dev/phii-identity/hpc/b14"

# name -> patch (None = the shipped scheme, unpatched)
SCHEMES = {
    "c3c": None,
    "o1": ARMS / "k10_o1_arm.patch",
    "k11": ARMS / "k11_arm.patch",
}

# A dense-cloud ambient and a real cluster's ionising output, matching
# test_phii_c3c_spitzer.py so the two files cannot drift apart on the anchor.
N0_CGS = 1e3
QI_AU = 5.1227849481751455e64
TOL = 1e-12


# =============================================================================
# Scheme loading
# =============================================================================

@lru_cache(maxsize=None)
def load_scheme(name):
    """Return the scheme's `get_phii_c3c` entry point.

    Every call site in trinity reaches the closure through that one name, and both
    arm patches alias themselves onto it (`get_phii_c3c = get_phii_k11`), so this is
    the same function the phase runners would call under the arm.
    """
    patch = SCHEMES[name]
    if patch is None:
        mod_path = REPO / HELPER_REL
    else:
        if not patch.is_file():
            # The arm patches are untracked (a32b098e), so on a fresh clone this suite
            # used to report "6 passed, 15 skipped" and look green -- while the three
            # EXPECTED_FAIL assertions that ARE the structural result (O1 cannot pass
            # L1/L1b/L4) were never evaluated. A green suite that measured nothing is
            # worse than a red one. Fail loudly instead, with one deliberate opt-out.
            if os.environ.get("PHII_ALLOW_MISSING_ARMS") == "1":
                pytest.skip(f"{patch.name} absent, PHII_ALLOW_MISSING_ARMS=1 -- "
                            "shipped scheme only, structural result NOT measured")
            pytest.fail(
                f"{patch.name} is MISSING, so the '{name}' arm was not built and this "
                "suite's structural result was NOT measured. The patches live in "
                "docs/dev/phii-identity/hpc/b14/, untracked since a32b098e -- so a clean "
                "clone cannot run them. To test the shipped scheme alone on purpose, set "
                "PHII_ALLOW_MISSING_ARMS=1.",
                pytrace=False)
        tmp = Path(tempfile.mkdtemp(prefix=f"phii_{name}_"))
        dest = tmp / HELPER_REL
        dest.parent.mkdir(parents=True)
        shutil.copy2(REPO / HELPER_REL, dest)
        subprocess.run(["git", "apply", "-p1", str(patch)], cwd=tmp, check=True)
        mod_path = dest
    spec = importlib.util.spec_from_file_location(f"_phii_{name}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_phii_c3c


class _Shell:
    """Stand-in for ShellProperties. C3c reads only the absorbed fraction; both arms
    additionally read R_IF, and return 0.0 if it is missing -- which is exactly why the
    committed spitzer fixture (no R_IF) could not test them at all."""

    def __init__(self, R_IF=None, f_abs=1.0):
        self.shell_fAbsorbedIon = f_abs
        if R_IF is not None:
            self.R_IF = float(R_IF)


@lru_cache(maxsize=None)
def _base_params():
    """One ParameterDict, reused and mutated by every gate.

    ponytail: deliberately NOT deep-copied per call. `state()` below assigns every key
    any gate depends on, so a gate cannot inherit another's value -- verified, not
    assumed, by running the whole gate list forwards and then backwards and comparing
    all 21 results (identical). If a future gate reads a key `state()` does not set,
    copy here instead of debugging an order-dependent failure.
    """
    return read_param(str(REPO / "param" / "simple_cluster.param"))


def state(R2, Qi=QI_AU, Lmech=0.0, v_mech=2000.0, phase="momentum"):
    """A momentum-phase state, built the way run_momentum_phase.py builds one.

    The momentum phase is the right place for every limit here: there
    `get_effective_bubble_pressure` returns pRam directly, so P_conf is an explicit
    function of (Lmech, v_mech, R2) with no Eb or R1 root-solve in the way, and the
    runner assigns params['Pb'] = pRam so the shipped scheme's branch test compares
    against the same quantity the arms use. Wind strength is therefore one knob.
    """
    import trinity.bubble_structure.get_bubbleParams as gbp
    p = _base_params()
    P_ram = gbp.pRam(R2, Lmech, v_mech) if Lmech > 0 else 0.0
    for key, val in (("R2", R2), ("Qi", Qi), ("Pb", P_ram), ("current_phase", phase),
                     ("Lmech_total", Lmech), ("v_mech_total", v_mech),
                     ("Eb", 0.0), ("t_now", 1.0), ("tSF", 0.0)):
        p[key].value = val
    return p, P_ram


def stromgren_radius(p, Qi, n0):
    denom = 4.0 * np.pi * p["chi_e_shell"].value * p["caseB_alpha"].value * n0**2
    return (3.0 * Qi / denom) ** (1.0 / 3.0)


def p_ref(p):
    """(mu_c/mu_i) k_B T -- the pressure of ionised gas per hydrogen nucleon."""
    return (p["mu_convert"].value / p["mu_ion_shell"].value
            * p["k_B"].value * p["TShell_ion"].value)


def compose(phase, ret, P_conf, P_ram):
    """P_drive as the four phase runners actually build it (PLAN.md, composition table).

    A helper's return value is meaningless on its own: the arms return the drive MINUS
    whatever the phase's own expression adds back. Every gate that claims something
    about the driving pressure must go through this.
    """
    if phase in ("energy", "implicit"):
        return max(P_conf, ret)
    if phase == "transition":
        return max(P_conf, ret + P_ram)
    return ret + P_ram


# =============================================================================
# The gates. Each returns a dict; the tests assert on it and __main__ writes it out.
# =============================================================================

def gate_L1(name):
    """Photoionisation-only: wind off, front at R2, drive must be the Stromgren value."""
    phii = load_scheme(name)
    p, _ = state(R2=1.0, Lmech=0.0)          # P_conf = P_ram = 0
    n0 = N0_CGS * __import__("trinity._functions.unit_conversions",
                             fromlist=["x"]).ndens_cgs2au
    R_St = stromgren_radius(p, QI_AU, n0)
    p["R2"].value = R_St
    got = phii(p, _Shell(R_IF=R_St))
    drive = compose("momentum", got, 0.0, 0.0)
    want = p_ref(p) * n0                      # n_tot k T at the Stromgren radius
    ok = want > 0 and abs(drive / want - 1.0) < 1e-10
    return dict(gate="L1", scheme=name, passed=ok, measured=drive, expected=want,
                rel=(drive / want - 1.0) if want else float("nan"),
                note="drive at R_St with the wind off; want p_ref*n0 = rho0*c_i^2")


def gate_L1b(name):
    """The same limit's SCALING: P ~ Qi^(1/2) R^(-3/2) is what makes it Stromgren."""
    phii = load_scheme(name)
    vals = []
    for R2 in (1.0, 2.0, 4.0, 8.0):
        p, _ = state(R2=R2, Lmech=0.0)
        vals.append(compose("momentum", phii(p, _Shell(R_IF=R2)), 0.0, 0.0))
    if min(vals) <= 0:
        return dict(gate="L1b", scheme=name, passed=False, measured=0.0, expected=-1.5,
                    rel=float("nan"), note="no photo-only drive to fit a slope to")
    slope = np.polyfit(np.log([1.0, 2.0, 4.0, 8.0]), np.log(vals), 1)[0]
    return dict(gate="L1b", scheme=name, passed=abs(slope + 1.5) < 1e-6, measured=slope,
                expected=-1.5, rel=slope + 1.5, note="d log P / d log R2 at fixed Qi")


def gate_L2(name):
    """Wind-only: as photons -> 0 the COMPOSED momentum drive must tend to P_ram.

    ponytail: this gate has now been caught passing for the wrong reason TWICE, by two
    different audits, and both are recorded here because the pattern is the point.
      (1) As first written it used Qi = 0 exactly -- where all three schemes return
          through the shared `if not (R2 > 0 and Qi > 0)` guard and no closure runs.
      (2) Rewritten as a limit, it still could not fail for two of the three: with the
          front at R_IF = R2 (the physical wind-only geometry -- no photons, no ionised
          layer) O1's momentum return is pRam*(rho-1) with rho == 1 EXACTLY, so it is
          identically 0 for any Qi whatsoever, and C3c returns 0 through its confined
          branch. Only K11 actually converges.
    Both legs are kept, because the first IS the physical limit and a scheme that meets
    it structurally genuinely meets it -- but the verdict now says WHICH, and leg (b)
    holds the front open so the closure's Qi-dependence is exercised rather than
    geometrically annihilated.
    """
    phii = load_scheme(name)
    seq = []
    for fq in (1e-10, 1e-20):
        p, P_ram = state(R2=5.0, Qi=QI_AU * fq, Lmech=1.0e5)
        seq.append(compose("momentum", phii(p, _Shell(R_IF=5.0, f_abs=1.0)), P_ram, P_ram) / P_ram)
    # leg (b): front held open, so rho != 1 and the return is not annihilated by geometry
    openf = []
    for fq in (1.0, 1e-10, 1e-20):
        p, P_ram = state(R2=5.0, Qi=QI_AU * fq, Lmech=1.0e5)
        openf.append(compose("momentum", phii(p, _Shell(R_IF=5.25, f_abs=1.0)), P_ram, P_ram) / P_ram)
    trivial = abs(seq[0] - 1.0) == 0.0 and abs(seq[-1] - 1.0) == 0.0
    ok = abs(seq[-1] - 1.0) < TOL and abs(seq[-1] - 1.0) <= abs(seq[0] - 1.0)
    how = ("satisfied STRUCTURALLY (the return is identically 0 at R_IF = R2, for any Qi) "
           if trivial else "satisfied by CONVERGENCE")
    return dict(gate="L2", scheme=name, passed=ok, measured=seq[-1], expected=1.0,
                rel=seq[-1] - 1.0,
                note=f"{how}; R_IF=R2 leg at Qi x (1e-10, 1e-20) = "
                     + ", ".join(f"{v:.12f}" for v in seq)
                     + f"; front-held-open leg (R_IF/R2 = 1.05) at Qi x (1, 1e-10, 1e-20) = "
                     + ", ".join(f"{v:.6f}" for v in openf)
                     + " -- a flat sequence there means the drive carries no photon "
                       "dependence at fixed front, which is L1's finding seen sideways")


def gate_L3(name):
    """Strong-wind confined fork -- RECORDED, NOT SCORED. This IS D5's shape.

    Swept rather than sampled at one point, because the fork only opens as the photon
    term is driven under the wind term: at the asymptote C3c transmits (ratio 1, it
    returns 0), O1 amplifies by the front's area (ratio (R_IF/R2)^2), and K11 tends to
    P_ram with NO amplification because its self-consistent skin puts the wind contact
    at the front. Which of those trinity should adopt is D5.
    """
    phii = load_scheme(name)
    R2, amp = 5.0, 1.002
    out = []
    for f in (1e-6, 1e-9, 1e-12):
        p, P_ram = state(R2=R2, Qi=QI_AU * f, Lmech=1.0e7)
        out.append(compose("momentum", phii(p, _Shell(R_IF=R2 * amp)), P_ram, P_ram) / P_ram)
    return dict(gate="L3", scheme=name, passed=None, measured=out[-1], expected=float("nan"),
                rel=out[0] - out[-1],
                note=f"composed drive/P_ram at Qi x (1e-6, 1e-9, 1e-12) with R_IF/R2 = {amp}: "
                     + ", ".join(f"{v:.6f}" for v in out)
                     + f"; O1's area factor is {amp**2:.6f}, K11's asymptote is 1.0")


def gate_L4(name):
    """R_IF -> R2 collapse: K11 -> the shipped P_C3a, O1 -> P_conf, over many radii."""
    c3c = load_scheme("c3c")
    phii = load_scheme(name)
    worst = 0.0
    for R2 in np.logspace(-2, 2, 40):
        p, _ = state(R2=R2, Lmech=0.0)
        ref = compose("momentum", c3c(p, _Shell(R_IF=R2)), 0.0, 0.0)
        got = compose("momentum", phii(p, _Shell(R_IF=R2)), 0.0, 0.0)
        if ref > 0:
            worst = max(worst, abs(got / ref - 1.0))
    return dict(gate="L4", scheme=name, passed=worst < TOL, measured=worst, expected=0.0,
                rel=worst, note="max |drive/P_C3a - 1| over 40 radii, wind off, R_IF = R2")


def gate_L5(name):
    """Degenerate inputs return exactly 0.0 -- never NaN, never inf.

    ponytail: the first version listed three "cases" (R2<0, Qi=0, R2=0) that all tripped
    the SAME first guard `if not (R2 > 0 and Qi > 0)`, so it was one test relabelled
    three times and the stated contract (never NaN/inf) was never exercised at all.
    Independent audit, 2026-09-02. The probes below are split into what is ASSERTED
    (finite, physically reachable degeneracies) and what is RECORDED -- because the
    recorded set found a real hole: the SHIPPED scheme returns `inf` for Qi = inf, where
    both arms return 0.0. Not reachable from an SPS table, so it is reported to the
    maintainer rather than asserted as a failure.
    """
    phii = load_scheme(name)
    bad = []
    # asserted: each of these must reach a *different* guard or a real evaluation
    p, _ = state(R2=-1.0, Lmech=0.0)
    if phii(p, _Shell(R_IF=1.0)) != 0.0:
        bad.append("R2<0")
    p, _ = state(R2=1.0, Qi=0.0, Lmech=0.0)
    if phii(p, _Shell(R_IF=1.0)) != 0.0:
        bad.append("Qi=0")
    p, _ = state(R2=1.0, Qi=-QI_AU, Lmech=0.0)
    if phii(p, _Shell(R_IF=1.0)) != 0.0:
        bad.append("Qi<0")
    p, _ = state(R2=1.0, Lmech=0.0)
    for label, shell in (("no R_IF", _Shell()), ("R_IF<R2", _Shell(R_IF=0.5))):
        out = phii(p, shell)
        if not math.isfinite(out):
            bad.append(f"{label}->{out!r}")
        if name != "c3c" and out != 0.0:
            bad.append(f"{label} not guarded ->{out!r}")
    # recorded, not asserted: the non-finite probes
    probes = {}
    for label, kw in (("Qi=inf", dict(Qi=float("inf"))), ("Qi=nan", dict(Qi=float("nan"))),
                      ("R2=nan", dict(R2=float("nan")))):
        p, _ = state(R2=kw.pop("R2", 1.0), Lmech=0.0, **kw)
        try:
            probes[label] = phii(p, _Shell(R_IF=1.0))
        except Exception as exc:                      # a raise is also a finding
            probes[label] = f"raised {type(exc).__name__}"
    leaky = [f"{k}->{v!r}" for k, v in probes.items()
             if isinstance(v, float) and not math.isfinite(v)]
    return dict(gate="L5", scheme=name, passed=not bad, measured=len(bad), expected=0,
                rel=float(len(leaky)),
                note=("; ".join(bad) if bad else "all reachable degeneracies -> 0.0, finite")
                     + " | RECORDED non-finite probes: "
                     + ", ".join(f"{k}={v!r}" for k, v in probes.items())
                     + (f" | ⛔ {len(leaky)} leak non-finite: {', '.join(leaky)}" if leaky
                        else " | none leak"))


def gate_L6(name):
    """Sign: the stored P_HII must not go negative, in momentum OR transition.

    THE BOUND, and the correction to how it was first stated here. In exact arithmetic
    it is provable for all three:
      C3c  returns 0.0 or P_C3a > 0.
      O1   drive = P_conf (R_IF/R2)^2 with the helper guarding R_IF >= R2, so
           drive >= P_conf; transition P_conf = max(P_thermal, P_ram) >= P_ram.
      K11  n >= n_w = n0 (R2/R_IF)^2 (the patch's bracket proof), so
           drive = p_ref n (R_IF/R2)^2 >= p_ref n0 = P_conf >= P_ram.
    ⛔ In FLOAT it is not exact for K11. An independent audit (2026-09-02) brute-forced
    ~98k states per scheme and found C3c and O1 clean but K11 negative on ~6.6% of them,
    worst ret/P_ram = -5.6e-16. Cause: the floor identity p_ref n_w (R_IF/R2)^2 == P_ram
    is only good to ~8e-15 relative through the (..)**1.5 / (..)**(2/3) round trip in
    `_k11_skin_density`, so when the wind term dominates the photon term by ~1e22 the
    root lands just above n_w and `drive - pRam` cancels into a negative residue. That
    corner is not reachable in trinity's own rows (photon-dominated by 10-92x on driving
    rows), so this gate ASSERTS on the physical grid and RECORDS the leak separately
    rather than pretending either that the bound is exact or that the leak is dynamics.
    The first version of this gate could not see it: it fixed Qi, fixed R_IF/R2 = 1.4 and
    Eb = 0, so its whole grid sat 15x away from the boundary it claimed to bound.
    """
    phii = load_scheme(name)
    neg_m = neg_t = 0
    margin = float("inf")
    for R2 in np.logspace(-1, 1.5, 25):
        for Lmech in (1e3, 1e5, 1e7):
            p, P_ram = state(R2=R2, Lmech=Lmech)
            if phii(p, _Shell(R_IF=R2 * 1.4)) < 0:
                neg_m += 1
            p["current_phase"].value = "transition"   # P_conf = max(P_thermal, P_ram)
            ret_t = phii(p, _Shell(R_IF=R2 * 1.4))
            if ret_t < 0:
                neg_t += 1
            if P_ram > 0:
                margin = min(margin, (ret_t + P_ram) / P_ram)
    # measured by the gate, not quoted from a side probe: Eb > 0 only widens the margin
    wide = []
    for Eb in (1e5, 1e7):
        p, P_ram = state(R2=5.0, Lmech=1e5)
        p["Eb"].value = Eb
        p["current_phase"].value = "transition"
        wide.append((phii(p, _Shell(R_IF=7.0)) + P_ram) / P_ram)
    # RECORDED: the wind-dominated corner where the float identity leaks
    leak, worst = 0, 0.0
    for fq in (1e-20, 1e-30):
        for R2 in np.logspace(-1, 1.5, 10):
            p, P_ram = state(R2=R2, Qi=QI_AU * fq, Lmech=1e9)
            r = phii(p, _Shell(R_IF=R2 * 1.05))
            if r < 0:
                leak += 1
                worst = min(worst, r / P_ram)
    return dict(gate="L6", scheme=name, passed=(neg_m == 0 and neg_t == 0), measured=margin,
                expected=0, rel=float(neg_m + neg_t),
                note=f"{neg_m} negative momentum + {neg_t} negative transition returns of 75 "
                     f"each on the physical grid; min transition drive/P_ram = {margin:.6g} "
                     f"(O1's is exactly (R_IF/R2)^2 = 1.96); Eb = 1e5, 1e7 widen it to "
                     + ", ".join(f"{w:.4g}" for w in wide)
                     + f" | RECORDED wind-dominated corner (Qi x 1e-20, 1e-30, Lw 1e9): "
                       f"{leak}/20 negative, worst ret/P_ram = {worst:.3e} "
                       f"({'roundoff on the float floor identity' if leak else 'clean'})")


GATES = [gate_L1, gate_L1b, gate_L2, gate_L3, gate_L4, gate_L5, gate_L6]

# Gates a scheme is EXPECTED to fail, with the reason. An expected failure that starts
# passing is as much a finding as a pass that starts failing, so both are asserted.
EXPECTED_FAIL = {
    ("o1", "L1"): "structural: drive is proportional to P_conf, so wind -> 0 kills it",
    ("o1", "L1b"): "structural: no photo-only drive exists to have a slope",
    ("o1", "L4"): "structural: O1 reduces to P_conf, not to P_C3a",
}


# =============================================================================
# pytest
# =============================================================================

@pytest.mark.parametrize("scheme", list(SCHEMES))
@pytest.mark.parametrize("gate", GATES, ids=[g.__name__[5:] for g in GATES])
def test_limit(scheme, gate):
    r = gate(scheme)
    if r["passed"] is None:
        pytest.skip(f"{r['gate']} is recorded, not scored: {r['note']}")
    key = (scheme, r["gate"])
    if key in EXPECTED_FAIL:
        assert not r["passed"], (
            f"{scheme} now PASSES {r['gate']}, which it is not supposed to be able to: "
            f"{EXPECTED_FAIL[key]}. If the scheme changed, move this out of EXPECTED_FAIL "
            f"and say so in PLAN.md -- it would retire the workstream's structural result.")
    else:
        assert r["passed"], f"{scheme} FAILED {r['gate']}: {r['note']} (measured {r['measured']!r})"


# =============================================================================
# CSV emitter (C-6: stamped, so the artifact can be trusted)
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    missing = []
    for name in SCHEMES:
        if SCHEMES[name] is not None and not SCHEMES[name].is_file():
            print(f"  {name}: ** PATCH ABSENT -- ARM NOT MEASURED ** ({SCHEMES[name]})")
            missing.append(name)
            continue
        for gate in GATES:
            r = gate(name)
            r["expected_fail"] = EXPECTED_FAIL.get((name, r["gate"]), "")
            rows.append(r)
            verdict = ("RECORDED" if r["passed"] is None
                       else "pass" if r["passed"]
                       else ("FAIL (expected)" if (name, r["gate"]) in EXPECTED_FAIL else "FAIL"))
            print(f"  {name:4} {r['gate']:4} {verdict:16} {r['measured']!r}")
    if missing:
        print(f"\n  !! {len(missing)} arm(s) not measured: {', '.join(missing)}. "
              f"data/b23_limits.csv will be INCOMPLETE and the structural result "
              f"(O1 fails L1/L1b/L4) is NOT in it.\n")
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "unknown"
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        fh.write(f"# generated {now} | builder test_phii_limits.py | "
                 f"code {sha}{'+dirty' if dirty else ''}\n")
        fh.write("# Limits SSOT: every P_HII scheme against the same gates (PLAN.md 0.0 Phase 2).\n")
        fh.write("# passed='' means RECORDED-not-scored (L3, the D5 fork). expected_fail is the\n")
        fh.write("# reason a failure is structural rather than a defect.\n")
        w = csv.DictWriter(fh, fieldnames=["gate", "scheme", "passed", "measured", "expected",
                                           "rel", "expected_fail", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
