#!/usr/bin/env python3
"""H3 (Eb-floor) DIAGNOSTIC variant for the transition-trigger pt4 experiment.

>>> ENERGY-INJECTION CAVEAT — READ THIS FIRST <<<
The EBFLOOR variant FLOORS the bubble thermal energy Eb to a small POSITIVE
value so it can never collapse through zero. This INJECTS ENERGY and VIOLATES
energy conservation. It is a DIAGNOSTIC experiment ONLY, NOT a production fix
candidate. Its sole purpose: isolate whether "Eb going non-positive" is the
SOLE failure mode for the massive-cloud collapse regime, i.e. whether the rest
of the physics stays intact once Eb is forced > 0.

Production is NEVER touched. The patch monkeypatches module attributes only;
call `apply(variant, floor=...)` BEFORE running the sim (one sim per process).

Pattern copied from docs/dev/failed-large-clouds/harness/variants.py
(do NOT edit that original). V0/V1/V2/V3 carried over verbatim so this harness
is self-contained; EBFLOOR is the new piece.

Variants
--------
V0       baseline               : no patch (reference).
V1       solve_R1 R1-clamp      : clamp R1 <= R2*(1-eps).
V2       bubble_E2P vol-floor   : floor shell volume (r2^3-r1^3) at eps*r2^3.
V3       both                   : V1 + V2 (the prior "geometry-guard-only" smoke).
EBFLOOR  positive-Eb floor      : clamp the *state* Eb > 0 everywhere. Two
                                  coupled pieces:

  (A) DRIVE stays positive. Clamp Eb = max(Eb, floor) at entry to
      get_bubbleParams.bubble_E2P (-> Pb > 0) and .solve_R1 (-> R1 well-defined),
      then DELEGATE to the ORIGINAL production helpers (which already carry the
      shipped geometry guards: E2P floors shell_volume at 1e-13*r2**3 :229-235;
      solve_R1 returns 0 for R2<=0 :433). So when Eb >= floor this is BIT-
      IDENTICAL to V0. Reached everywhere via the module attribute (phase 1a/1b
      RHS, bubble solve, diagnostics). Keeps the shell push positive even as the
      state Eb decays toward / below the floor.

  (B) STATE stays positive (the key new piece). The shipped ENERGY_COLLAPSED
      early-stops read the ODE STATE Eb extracted from solution.y[:,-1]
      (run_energy_phase.py:340, run_energy_implicit_phase.py:1007). To keep the
      run going we must keep that *integrated state* >= floor. We install a
      REFLECTING FLOOR on dEb/dt: whenever Eb <= floor, the energy-derivative
      component returned to solve_ivp is clamped to be >= 0, so the state can
      only hold flat or grow at the floor and never integrates below it.
      Implemented by wrapping the two RHS functions each phase's solve_ivp uses:
        - phase 1a: energy_phase_ODEs.get_ODE_Edot_pure (3rd return = Ed)
        - phase 1b: run_energy_implicit_phase.get_ODE_implicit_pure (3rd return =
          Ed_from_beta, the betadelta cooling derivative actually integrated).
      Both are patched in the module namespace each solve_ivp closure reads from.

  WHAT (B) BYPASSES: the shipped Eb<=0 -> ENERGY_COLLAPSED clean stops in both
  energy phases. By construction the state never reaches <= 0, so those guards
  never fire. That is the whole point of the experiment — and exactly why this
  is NOT a production fix (it manufactures energy to keep the bubble alive).

Floor choice: a small positive value [au] = Msun*pc**2/Myr**2. Default
FLOOR = 1e-3 (tiny vs the collapse regime's E0 ~ 1e10-1e12; sensitivity noted
in the writeup). apply(..., floor=X) overrides. Single process => a module
global is safe (ponytail: no concurrency).
"""
import numpy as np

import trinity.bubble_structure.get_bubbleParams as gbp
import trinity._functions.unit_conversions as cvt
import trinity.phase1_energy.energy_phase_ODEs as epo
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rimp

EPS = 1e-6
FLOOR = 1e-3  # [au] Eb floor; set by apply(floor=...). Reflecting on the state.

# activation telemetry: did the floor ever bite? (set by the patched callables)
ACTIVATED = {"drive": 0, "state": 0, "min_Eb_seen": float("inf")}

_ORIG_SOLVE_R1 = gbp.solve_R1
_ORIG_E2P = gbp.bubble_E2P
_ORIG_EDOT = epo.get_ODE_Edot_pure
_ORIG_IMPLICIT = rimp.get_ODE_implicit_pure


# ---- V1/V2/V3 geometry guards (verbatim from the flc harness) ---------------
def _solve_R1_clamped(R2, Eb, Lmech_total, v_mech_total):
    R1 = _ORIG_SOLVE_R1(R2, Eb, Lmech_total, v_mech_total)
    R1_max = R2 * (1.0 - EPS)
    if R1 > R1_max:
        return R1_max
    return R1


def _bubble_E2P_vol_floored(Eb, r2, r1, gamma):
    r1c = r1 * cvt.pc2cm
    r2c = r2 * cvt.pc2cm
    Ebc = Eb * cvt.E_au2cgs
    vol = r2c**3 - r1c**3
    floor = EPS * r2c**3
    if vol < floor:
        vol = floor
    Pb = (gamma - 1) * Ebc / vol / (4 * np.pi / 3)
    return Pb * cvt.Pb_cgs2au


# ---- EBFLOOR (A): clamp Eb in the drive helpers -----------------------------
# Delegate to the ORIGINAL production helpers (which already carry the shipped
# geometry guards: bubble_E2P floors shell_volume at 1e-13*r2**3 [get_bubbleParams
# .py:229-235]; solve_R1 returns 0 for R2<=0 [:433]). So when Eb >= FLOOR these
# are BIT-IDENTICAL to V0 -- the only change vs production is clamping Eb. That
# makes EBFLOOR a provable no-op on any run whose Eb never drops below FLOOR.
def _solve_R1_ebfloor(R2, Eb, Lmech_total, v_mech_total):
    if Eb < FLOOR:
        ACTIVATED["drive"] += 1
        ACTIVATED["min_Eb_seen"] = min(ACTIVATED["min_Eb_seen"], Eb)
        Eb = FLOOR
    return _ORIG_SOLVE_R1(R2, Eb, Lmech_total, v_mech_total)


def _bubble_E2P_ebfloor(Eb, r2, r1, gamma):
    if Eb < FLOOR:
        ACTIVATED["drive"] += 1
        ACTIVATED["min_Eb_seen"] = min(ACTIVATED["min_Eb_seen"], Eb)
        Eb = FLOOR
    return _ORIG_E2P(Eb, r2, r1, gamma)


# ---- EBFLOOR (B): reflecting floor on the integrated state Eb ---------------
def _edot_ebfloor(t, y, snapshot, params_for_feedback):
    """Phase-1a RHS wrapper: clamp dEb/dt >= 0 when the state Eb is at/below the
    floor, so solve_ivp can never integrate the state below FLOOR."""
    rd, vd, Ed = _ORIG_EDOT(t, y, snapshot, params_for_feedback)
    Eb = y[2]
    if Eb <= FLOOR:
        ACTIVATED["state"] += 1
        ACTIVATED["min_Eb_seen"] = min(ACTIVATED["min_Eb_seen"], Eb)
        if Ed < 0.0:
            Ed = 0.0
    return [rd, vd, Ed]


def _implicit_ebfloor(t, y, snapshot, params_for_feedback, Ed_from_beta, Td_from_delta):
    """Phase-1b RHS wrapper: same reflecting floor on the betadelta energy
    derivative actually integrated by the implicit-phase solve_ivp."""
    Eb = y[2]
    if Eb <= FLOOR:
        ACTIVATED["state"] += 1
        ACTIVATED["min_Eb_seen"] = min(ACTIVATED["min_Eb_seen"], Eb)
        if Ed_from_beta < 0.0:
            Ed_from_beta = 0.0
    return _ORIG_IMPLICIT(t, y, snapshot, params_for_feedback, Ed_from_beta, Td_from_delta)


def _restore():
    gbp.solve_R1 = _ORIG_SOLVE_R1
    gbp.bubble_E2P = _ORIG_E2P
    epo.get_ODE_Edot_pure = _ORIG_EDOT
    rimp.get_ODE_implicit_pure = _ORIG_IMPLICIT


def apply(variant: str, floor: float | None = None):
    """Install the named variant's monkeypatch. Returns the variant id."""
    global FLOOR
    _restore()  # idempotent
    ACTIVATED["drive"] = 0
    ACTIVATED["state"] = 0
    ACTIVATED["min_Eb_seen"] = float("inf")
    if floor is not None:
        FLOOR = float(floor)
    if variant == "V0":
        pass
    elif variant == "V1":
        gbp.solve_R1 = _solve_R1_clamped
    elif variant == "V2":
        gbp.bubble_E2P = _bubble_E2P_vol_floored
    elif variant == "V3":
        gbp.solve_R1 = _solve_R1_clamped
        gbp.bubble_E2P = _bubble_E2P_vol_floored
    elif variant == "EBFLOOR":
        # (A) drive helpers clamp Eb then delegate to the ORIGINAL production
        #     helpers (shipped geometry guards already inside). Bit-identical to
        #     V0 when Eb >= floor.
        gbp.solve_R1 = _solve_R1_ebfloor
        gbp.bubble_E2P = _bubble_E2P_ebfloor
        # (B) reflecting floor on the integrated state, both phases.
        epo.get_ODE_Edot_pure = _edot_ebfloor
        rimp.get_ODE_implicit_pure = _implicit_ebfloor
    else:
        raise ValueError(f"unknown variant {variant!r} (V0/V1/V2/V3/EBFLOOR)")
    return variant
