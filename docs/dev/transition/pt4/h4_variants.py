#!/usr/bin/env python3
"""H4 (PdV-drain cap) DIAGNOSTIC variant for the transition-trigger pt4 experiment.

>>> ENERGY-INJECTION CAVEAT — READ THIS FIRST <<<
The PDVCAP variant CAPS the PdV expansion-work drain in dEb/dt at `kappa*Lmech`
for an early window `t < t_window`. The PdV term `4*pi*R2**2*Pb*v2` is REAL
PHYSICAL expansion work (here Lcool/L_bubble is only ~1% of Lmech; the bubble is
drained by doing work on the shell, not by radiating). Capping it therefore
UNDER-DRAINS the bubble / INJECTS ENERGY and VIOLATES energy conservation while
the cap is active. This is a DIAGNOSTIC experiment ONLY, NOT a production fix
candidate. Its sole purpose: answer whether the massive-cloud bubble, given a
brief reprieve from the PdV drain, can ESTABLISH itself and SELF-SUSTAIN after
the cap lifts (survivable early transient), or RE-COLLAPSES the moment the cap
is released because PdV>Lmech persists ("stillborn" / momentum-driven from
birth, per failed-large-clouds PLAN.md §3b).

Production is NEVER touched. The patch monkeypatches module attributes only;
call `apply(variant, t_window=..., kappa=...)` BEFORE running the sim
(one sim per process — trinity leaks module-level globals in-process).

Pattern copied from the H3 sibling (h3_variants.py): same monkeypatch-of-module-
attributes approach, same idempotent restore, same activation telemetry. The
recomputed PdV uses the production bubble_E2P, whose shipped shell-volume floor
(1e-13*r2^3, get_bubbleParams.py:229-235) keeps the divide finite as Eb -> 0.
V0 (baseline, no patch) carried over verbatim.

-------------------------------------------------------------------------------
THE INJECTION POINT — and the proof the DRIVE/shell-acceleration is UNTOUCHED
-------------------------------------------------------------------------------
The PdV drain shares `press_bubble` with the shell drive, so a naive
`get_effective_bubble_pressure` patch would ALSO throttle the shell push (the
bubble would stop expanding) — forbidden. We therefore wrap the ODE RHS and
modify ONLY the energy derivative, never the (rd, vd) shell-acceleration terms.

Strategy: "add back the capped excess." We delegate the WHOLE RHS to the
ORIGINAL production function, take its returned derivatives, and adjust ONLY the
energy component:
    PdV    = 4*pi*R2**2 * press_bubble * v2     [same Pb,R2,v2 the original used]
    cap    = kappa * Lmech_total
    excess = max(0.0, PdV - cap)                 [>0 only when the drain exceeds the cap]
    Ed_capped = Ed_orig + excess                 [== replacing PdV with min(PdV,cap)]
Because we ADD `excess` to the original Ed, and `excess=0` whenever PdV<=cap or
t>=t_window, the variant is BIT-IDENTICAL to V0 on any step where the cap does
not bite. The returned rd (=v2) and vd (shell acceleration, built from P_drive at
energy_phase_ODEs.py:265-266, which does NOT use the PdV term) are passed through
UNCHANGED — the shell keeps expanding exactly as in production.

PHASE 1a (energy_phase_ODEs.get_ODE_Edot_pure, line 280:
    Ed = (Lmech_total - L_bubble) - (4*pi*R2**2*press_bubble)*v2 - L_leak
the middle term is the PdV drain; vd at :265-266 uses P_drive=max(Pb,P_HII), a
separate quantity). We recompute `press_bubble` and `Lmech_total` with the SAME
helper calls the original makes (get_current_sps_feedback -> solve_R1 ->
get_effective_bubble_pressure; lines 195,223,226-231) so PdV matches the
original's drain term to machine precision, then add back the excess.
Called via the module attribute energy_phase_ODEs.get_ODE_Edot_pure
(run_energy_phase.py:272) — patchable.

PHASE 1b (run_energy_implicit_phase.get_ODE_implicit_pure, line 532 returns
[rd, vd, Ed_from_beta, Td_from_delta]). The 1b energy derivative actually
integrated is `Ed_from_beta` (Rahner-A12 cooling form, computed OUTSIDE the ODE
at run_energy_implicit_phase.py:854 and passed in at :929). PdV does NOT appear
as a separable term in Ed_from_beta; it enters the 1b physics through the
energy-balance the beta-solver matches:
    Edot_from_balance = L_gain - L_loss - 4*pi*R2**2*v2*Pb   (get_betadelta.py:434)
and the solver drives Ed_from_beta -> Edot_from_balance (residual :438). So the
faithful 1b analog of the 1a cap is to add the SAME capped-PdV excess back to
Ed_from_beta: Ed_from_beta + max(0, PdV - kappa*Lmech). rd, vd are again passed
through untouched (the wrapper modifies only the 3rd component before delegating).

Activation telemetry (set by the patched callables): how many RHS evals the cap
bit, the max PdV/Lmech seen, and the cumulative-injected-energy proxy. A run
whose PdV never exceeds the cap (healthy/stall: PdV<Lmech<cap) is a provable
no-op == V0.
"""

import numpy as np

import trinity.bubble_structure.get_bubbleParams as gbp
import trinity.phase1_energy.energy_phase_ODEs as epo
import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rimp

# cap parameters; set by apply(...). Single process => module globals are safe
# (ponytail: no concurrency — one sim per process by construction).
T_WINDOW = 1e-3  # [Myr] cap active while t < T_WINDOW; after that, production behavior.
KAPPA = 0.9  # cap PdV drain at KAPPA*Lmech_total during the window.

# activation telemetry: did the cap ever bite, and by how much?
ACT = {
    "n1a": 0,
    "n1b": 0,  # RHS evals where the cap bit (1a / 1b)
    "max_pdv_ratio": 0.0,  # max PdV/Lmech the RHS saw (any t)
    "max_pdv_ratio_in_window": 0.0,  # max PdV/Lmech during t < t_window
    "max_pdv_ratio_after": 0.0,  # max PdV/Lmech at t >= t_window (post-cap)
    "sum_excess": 0.0,  # cumulative added energy-rate (proxy; counts trial steps)
}

_ORIG_EDOT = epo.get_ODE_Edot_pure
_ORIG_IMPLICIT = rimp.get_ODE_implicit_pure


def _press_bubble(t, R2, Eb, snapshot, params):
    """Recompute (press_bubble, Lmech_total) using the SAME production helpers
    the RHS uses (energy_phase_ODEs.py:195,223,226-231 for 1a;
    get_betadelta.compute_R1_Pb -> bubble_E2P for the 1b balance at
    get_betadelta.py:414,434). get_effective_bubble_pressure's energy/implicit
    branch calls the production bubble_E2P, which already carries the shipped
    shell-volume floor 1e-13*r2^3 (get_bubbleParams.py:229-235), so the PdV
    divide cannot blow up as Eb -> 0 / R1 -> R2 — no extra guard needed.
    Returns (Pb, Lmech_total) — matching the original drain term to machine
    precision wherever the production code path matches."""
    feedback = epo.get_current_sps_feedback(t, params)
    Lmech_total = feedback.Lmech_total
    v_mech_total = feedback.v_mech_total
    R1 = gbp.solve_R1(R2, Eb, Lmech_total, v_mech_total)
    Pb = gbp.get_effective_bubble_pressure(
        current_phase=snapshot.current_phase,
        Eb=Eb,
        R2=R2,
        R1=R1,
        gamma=snapshot.gamma_adia,
        Lmech_total=Lmech_total,
        v_mech_total=v_mech_total,
        t=t,
        tSF=snapshot.tSF,
    )
    return Pb, Lmech_total


# ---- PDVCAP (A): phase 1a wrapper -------------------------------------------
def _edot_pdvcap(t, y, snapshot, params_for_feedback):
    rd, vd, Ed = _ORIG_EDOT(t, y, snapshot, params_for_feedback)
    R2, v2, Eb = y
    Pb, Lmech = _press_bubble(t, R2, Eb, snapshot, params_for_feedback)
    PdV = 4.0 * np.pi * R2**2 * Pb * v2
    if Lmech > 0:
        ratio = PdV / Lmech
        ACT["max_pdv_ratio"] = max(ACT["max_pdv_ratio"], ratio)
        if t < T_WINDOW:
            ACT["max_pdv_ratio_in_window"] = max(ACT["max_pdv_ratio_in_window"], ratio)
        else:
            ACT["max_pdv_ratio_after"] = max(ACT["max_pdv_ratio_after"], ratio)
    if t < T_WINDOW:
        cap = KAPPA * Lmech
        excess = PdV - cap
        if excess > 0.0:
            ACT["n1a"] += 1
            ACT["sum_excess"] += excess
            Ed = Ed + excess  # == replacing PdV with min(PdV, cap) in the drain
    return [rd, vd, Ed]


# ---- PDVCAP (B): phase 1b wrapper -------------------------------------------
def _implicit_pdvcap(t, y, snapshot, params_for_feedback, Ed_from_beta, Td_from_delta):
    R2, v2, Eb, T0 = y
    if t < T_WINDOW:
        Pb, Lmech = _press_bubble(t, R2, Eb, snapshot, params_for_feedback)
        PdV = 4.0 * np.pi * R2**2 * Pb * v2
        if Lmech > 0:
            ratio = PdV / Lmech
            ACT["max_pdv_ratio"] = max(ACT["max_pdv_ratio"], ratio)
            ACT["max_pdv_ratio_in_window"] = max(ACT["max_pdv_ratio_in_window"], ratio)
        cap = KAPPA * Lmech
        excess = PdV - cap
        if excess > 0.0:
            ACT["n1b"] += 1
            ACT["sum_excess"] += excess
            Ed_from_beta = Ed_from_beta + excess
    else:
        # telemetry only (post-cap): record PdV/Lmech so we can SEE it stay >1.
        Pb, Lmech = _press_bubble(t, R2, Eb, snapshot, params_for_feedback)
        PdV = 4.0 * np.pi * R2**2 * Pb * v2
        if Lmech > 0:
            ratio = PdV / Lmech
            ACT["max_pdv_ratio"] = max(ACT["max_pdv_ratio"], ratio)
            ACT["max_pdv_ratio_after"] = max(ACT["max_pdv_ratio_after"], ratio)
    # delegate to the ORIGINAL with the (possibly) adjusted energy derivative;
    # rd, vd inside the original come from get_ODE_Edot_pure UNCHANGED.
    return _ORIG_IMPLICIT(t, y, snapshot, params_for_feedback, Ed_from_beta, Td_from_delta)


def _restore():
    epo.get_ODE_Edot_pure = _ORIG_EDOT
    rimp.get_ODE_implicit_pure = _ORIG_IMPLICIT


def apply(variant: str, t_window: float | None = None, kappa: float | None = None):
    """Install the named variant's monkeypatch. Returns the variant id.

    variant : 'V0' (baseline, no patch) | 'PDVCAP' (the cap).
    t_window [Myr], kappa [-] : cap parameters (override the module defaults).
    """
    global T_WINDOW, KAPPA
    _restore()  # idempotent
    for k in ACT:
        ACT[k] = 0 if k.startswith("n") else 0.0
    if t_window is not None:
        T_WINDOW = float(t_window)
    if kappa is not None:
        KAPPA = float(kappa)
    if variant == "V0":
        pass
    elif variant == "PDVCAP":
        epo.get_ODE_Edot_pure = _edot_pdvcap
        rimp.get_ODE_implicit_pure = _implicit_pdvcap
    else:
        raise ValueError(f"unknown variant {variant!r} (V0/PDVCAP)")
    return variant
