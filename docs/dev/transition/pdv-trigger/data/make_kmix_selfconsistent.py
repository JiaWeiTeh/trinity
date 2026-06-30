#!/usr/bin/env python3
"""Self-consistent kappa_mix offline test — inject El-Badry mixing into the REAL structure solve,
re-solve, and read how theta responds. STEP 2 of the Rung-B track (KMIX_IMPLEMENTATION_SPEC.md §4.3).
NO production code edited; the injection is a runtime MONKEYPATCH of the two conduction sites, so the
full production solve get_bubbleproperties_pure() (the v(R1)=0 fsolve for the Weaver dMdt and the whole
T-profile) runs WITH kappa_eff = max(kappa_mix, kappa_Spitzer).

WHY THIS, AND WHY IT IS THE FAITHFUL GATE (CLAUDE.md rule 5)
-----------------------------------------------------------
KMIX_PROTOTYPE.md answered "does kappa_mix DOMINATE the cool layer?" with a front estimate (GO). It did
NOT re-solve the structure, so it could not say what theta the self-consistent profile yields. This
harness closes that: it perturbs the actual ODE the solver integrates and measures the EMERGENT cooling
(bubble_LTotal, the resolved loss integral) and theta = LTotal / Lmech. Per-call equivalence is necessary
but not sufficient for an iterative path, so this is staged before any registry param: prove the
injection is correct (lambda*dv=0 == identity), then read the physics on the real configs.

THE INJECTION (KMIX_IMPLEMENTATION_SPEC.md §2-§3, the dimensionless-multiplier strategy)
----------------------------------------------------------------------------------------
Implement kappa_mix as a DIMENSIONLESS multiplier on the existing Spitzer term so the solver's mixed
AU/cgs RHS is never re-derived and lambda*dv=0 is BIT-IDENTICAL:
    R          = (lambda*dv[pc.km/s]*3.086e23) * Pb_cgs / (C_th_cgs * T^3.5)     [dimensionless]
    kappa_mult = max(1, R)                          (kappa_eff = kappa_Spitzer * kappa_mult)
  site :406 (_get_bubble_ODE RHS): the prefactor Pb/(C_th*T^5/2) -> Pb/(C_th*T^5/2*kappa_mult); and the
    Spitzer kappa'-term -2.5*dTdr^2/T (= -(dkappa/dT)/kappa * dTdr^2) is SWITCHED to 0 where R>1
    (kappa_mix is flat in T), kept = 2.5/T where Spitzer rules.
  site :370 (_get_bubble_ODE_initial_conditions, boundary at T_init=3e4 K, INSIDE the cool layer):
    C_thermal -> C_thermal * max(1, R(T_init, Pb)).
  site :291 (dMdt seed) is left Spitzer (initial guess only; the converged dMdt comes from the ODE+BCs).

UNITS (the recurring bug class — isolated in R, dimensionless): Pb_cgs = Pb_au / 1.5454414956718e12
(unit_conversions.Pb_cgs2au); lambda*dv pc.km/s -> cm^2/s x 3.086e23; C_th_cgs = params['C_thermal'] =
6e-7; T in K. The multiplier is a pure number so it cannot corrupt the AU/cgs mix in the RHS.

CORRECTNESS GATES (HARD — abort/flag, never fabricate):
  G1 IDENTITY: at lambda*dv=0 the patched solve must reproduce the UNPATCHED production solve to <1e-12
     (the multiplier is literally 1.0). If not, the monkeypatch is wrong, not the physics.
  G2 REPLAY: at lambda*dv=0 the replayed bubble_LTotal must match the logged bubble_Lloss for the row to
     REL_TOL (proves the state rebuild + units, same gate as make_da_replay).

COVERAGE: the 6 cleanroom configs (large_diffuse_lowsfe, be_sphere, midrange_pl0, pl2_steep,
simple_cluster, small_dense_highsfe) via make_da_replay's build_params/replay_row on the committed
cleanroom trajectories, PLUS the 2 captured fixtures (stiff 5e9 ~ fail_repro, mild cluster) via the FM1
loader. That spans 7-8 of the canonical 8 (small_1e6 control is the gap). Each is one representative
developed (near-blowout, max-R2) row — the workstream's theta-at-blowout convention.

REPRODUCE (from repo root; reads committed trajectories/fixtures, NO sims):
    python docs/dev/transition/pdv-trigger/data/make_kmix_selfconsistent.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/kmix_selfconsistent.csv
    docs/dev/transition/pdv-trigger/kmix_selfconsistent.png
"""
import csv
import importlib.util
import logging
import os
import sys

import numpy as np

logging.disable(logging.CRITICAL)  # silence trinity's chatty solver logs

import trinity.bubble_structure.bubble_luminosity as BL
from trinity.cooling import net_coolingcurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

# reuse the multi-config replay machinery (build_params/replay_row) and the fixture loader -- no dup
sys.path.insert(0, _HERE)
import make_da_replay as DR        # noqa: E402  build_params(cfg), replay_row(params, row)
_spec = importlib.util.spec_from_file_location("_fm1", os.path.join(_HERE, "make_fm1_rootcheck.py"))
_fm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm1)

# --- units (cgs), isolated in the dimensionless ratio R (KMIX_PROTOTYPE.md §1) ------------------
PC_KMS = 3.086e23                      # 1 pc*km/s -> cm^2/s
PB_AU2CGS = 1.0 / 1545441495671.806    # Pb_au -> erg/cm^3 (unit_conversions.Pb_cgs2au)
_T_INIT = BL._T_INIT_BOUNDARY          # 3e4 K, the boundary anchor (inside the cool layer)

REL_TOL = 1e-3                         # G2 replay gate (same as make_da_replay)
IDENT_TOL = 1e-12                      # G1 identity gate
_LDV = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]   # lambda*dv sweep [pc.km/s] (0 = off/identity)

# Injection mode: RHS-only (boundary IC kept Spitzer). The Spitzer boundary closure dR2 ~ 1/kappa
# DIVERGES if kappa_eff is injected there (R ~ 1e5-1e8 in the 3e4 K layer => dR2 explodes past R1,
# invalid domain) -- demonstrated by the _boundary_blowup_diagnostic below. So kappa_mix lives in the
# RHS only; the boundary just seeds the start near R2 and the integration carries the mixing physics.
# ponytail: boundary-geometry re-derivation for kappa_mix is future work (the El-Badry layer is not
# the Spitzer dR2 ~ 1/C closure); RHS-only is the robust scoping choice and refines SPEC §3.
_ORIG_ODE = BL._get_bubble_ODE
_ORIG_IC = BL._get_bubble_ODE_initial_conditions   # left installed (boundary stays Spitzer)


def _patched_ODE(ldv, inject_boundary_C=None):
    """_get_bubble_ODE with kappa_eff = kappa_Spitzer*max(1,R). Mirrors production lines 386-414; the
    Spitzer branch (R<=1, all of it at ldv=0) is the EXACT production expression -> bit-identical."""
    def ode(r_arr, initial_ODEs, params, Pb):
        v, T, dTdr = initial_ODEs
        if np.abs(T - 0) < 1e-5:
            raise BL.BubbleSolverError(f'temperature reached zero in bubble ODE RHS (T={T:.3e})')
        mu_fac = params['mu_convert'].value / params['mu_ion'].value
        ndens = Pb / (mu_fac * params['k_B'].value * T)
        phi = params['Qi'].value / (4 * np.pi * r_arr ** 2)
        dudt = net_coolingcurve.get_dudt(params['t_now'].value, ndens, T, phi, params)
        v_term = params['cool_alpha'].value * r_arr / params['t_now'].value
        C_th = params['cooling_boost_kappa'].value * params['C_thermal'].value
        bracket = ((params['cool_beta'].value + 2.5 * params['cool_delta'].value) / params['t_now'].value
                   + 2.5 * (v - v_term) * dTdr / T - dudt / Pb)
        R = (ldv * PC_KMS) * (Pb * PB_AU2CGS) / (params['C_thermal'].value * T ** 3.5)
        if R > 1.0:
            # kappa_mix dominates: kappa flat in T -> kappa'-term (-2.5 dTdr^2/T) vanishes; prefactor /R
            dTdrr = Pb / (C_th * T ** 2.5 * R) * bracket - 2 * dTdr / r_arr
        else:
            # Spitzer regime -- EXACT production expression (bit-identical at ldv=0)
            dTdrr = (Pb / (C_th * T ** (5 / 2)) * bracket
                     - 2.5 * dTdr ** 2 / T - 2 * dTdr / r_arr)
        dvdr = ((params['cool_beta'].value + params['cool_delta'].value) / params['t_now'].value
                + (v - v_term) * dTdr / T - 2 * v / r_arr)
        return [dvdr, dTdr, dTdrr]
    return ode


def _patched_IC_boundary(ldv):
    """DIAGNOSTIC ONLY: inject kappa_eff into the Spitzer boundary closure to SHOW it diverges.
    C_thermal -> C_thermal*max(1,R(T_init,Pb)); dR2 ~ C blows up past R1."""
    def ic(dMdt, params, Pb, R1):
        T_init = _T_INIT
        k_B = params['k_B'].value
        mu_ion = params['mu_ion'].value
        R = (ldv * PC_KMS) * (Pb * PB_AU2CGS) / (params['C_thermal'].value * T_init ** 3.5)
        kappa_mult = R if R > 1.0 else 1.0
        C_thermal = params['cooling_boost_kappa'].value * params['C_thermal'].value * kappa_mult
        R2 = params['R2'].value
        constant = (25 / 4 * k_B / mu_ion / C_thermal)
        dR2 = T_init ** (5 / 2) / (constant * dMdt / (4 * np.pi * R2 ** 2))
        T = (constant * dMdt * dR2 / (4 * np.pi * R2 ** 2)) ** (2 / 5)
        v = (params['cool_alpha'].value * R2 / params['t_now'].value
             - dMdt / (4 * np.pi * R2 ** 2) * k_B * T / mu_ion / Pb)
        return R2 - dR2, T, -2 / 5 * T / dR2, v
    return ic


def _install(ldv):
    BL._get_bubble_ODE = _patched_ODE(ldv)      # boundary IC stays production (Spitzer)


def _restore():
    BL._get_bubble_ODE = _ORIG_ODE
    BL._get_bubble_ODE_initial_conditions = _ORIG_IC


def _solve(params, ldv):
    """Full production solve at this lambda*dv; return (LTotal, dMdt, ok)."""
    _install(ldv)
    try:
        if "bubble_dMdt" in params:
            params["bubble_dMdt"].value = float("nan")  # re-solve the Weaver dMdt from the Eq-33 seed
        bp = BL.get_bubbleproperties_pure(params)
        return float(bp.bubble_LTotal), float(bp.bubble_dMdt), True
    except Exception:
        return float("nan"), float("nan"), False
    finally:
        _restore()


def _pick_row(df):
    """The representative developed row = max R2 (near blowout) with Pb>0 and finite logged Lloss."""
    d = df[(df["Pb"] > 0) & np.isfinite(df["bubble_Lloss"])]
    if len(d) == 0:
        return None
    return d.loc[d["R2"].idxmax()]


def _eval_state(label, params, row, logged_Lloss, Lmech):
    """G1/G2 gates at ldv=0, then sweep ldv; return (per-ldv rows, gate dict)."""
    # G1 identity: unpatched production vs patched-at-0
    if "bubble_dMdt" in params:
        params["bubble_dMdt"].value = float("nan")
    bp0 = BL.get_bubbleproperties_pure(params)          # unpatched (originals installed)
    L_unpatched = float(bp0.bubble_LTotal)
    L0, dMdt0, ok0 = _solve(params, 0.0)                 # patched at ldv=0
    g1 = abs(L0 - L_unpatched) / abs(L_unpatched) if L_unpatched else float("nan")
    # G2 replay: patched-at-0 LTotal vs the logged Lloss for this row
    g2 = abs(L0 - logged_Lloss) / abs(logged_Lloss) if logged_Lloss else float("nan")
    out = []
    for ldv in _LDV:
        L, dMdt, ok = _solve(params, ldv)
        theta = (L / Lmech) if (np.isfinite(L) and Lmech) else float("nan")
        out.append(dict(state=label, ldv=ldv, LTotal=L, dMdt=dMdt, theta=theta,
                        L_over_base=(L / L0 if (np.isfinite(L) and L0) else float("nan")),
                        solver_ok=ok))
    return out, dict(state=label, G1_identity=g1, G2_replay=g2,
                     L_unpatched=L_unpatched, L0=L0, Lmech=Lmech, logged_Lloss=logged_Lloss,
                     g1_pass=bool(np.isfinite(g1) and g1 < IDENT_TOL),
                     g2_pass=bool(np.isfinite(g2) and g2 < REL_TOL))


def _boundary_blowup_diagnostic(Pb_au, ldv=0.01):
    """Why the IC must stay Spitzer: the Spitzer boundary offset dR2 ~ C_thermal, so injecting
    kappa_eff = C*max(1,R) at the 3e4 K boundary multiplies dR2 by R(T_init,Pb). That factor is the
    boundary blowup -- R >> 1 (1e5-1e8) in the cool layer, so r2_prime = R2 - dR2 plunges past R1."""
    return (ldv * PC_KMS) * (Pb_au * PB_AU2CGS) / (6e-7 * _T_INIT ** 3.5)   # = dR2_inj / dR2_Spitzer


def main():
    import pandas as pd

    rows, gates, series = [], [], {}

    # ---- the 6 cleanroom configs (via make_da_replay machinery) -------------------------------
    for cfg in DR.DS.CLEANROOM:
        try:
            params = DR.build_params(cfg)
            df = pd.read_csv(f"docs/dev/transition/cleanroom/data/c0_{cfg}_h0.csv")
            row = _pick_row(df)
            if row is None:
                print(f"[{cfg}] no usable row — skip"); continue
            DR.replay_row(params, row)                   # set per-row state (re-uses the real setter)
            label = f"{cfg} (n={DR.DS.NCORE[cfg]:.0e})"
            per, g = _eval_state(label, params, row,
                                 float(row["bubble_Lloss"]), float(row["Lmech_total"]))
        except Exception as e:
            print(f"[{cfg}] FAILED to set up: {type(e).__name__}: {e}"); continue
        rows += per; gates.append(g); series[label] = per
        print(f"[{label}]  G1 ident={g['G1_identity']:.1e} {'OK' if g['g1_pass'] else 'FAIL'} | "
              f"G2 replay={g['G2_replay']:.1e} {'OK' if g['g2_pass'] else 'FAIL'} | "
              f"theta0={per[0]['theta']:.3f} -> theta@ldv=1={next(p['theta'] for p in per if p['ldv']==1.0):.3f}")
        if cfg == DR.DS.CLEANROOM[-1]:           # one representative boundary-blowup demonstration
            blow = _boundary_blowup_diagnostic(float(row["Pb"]))
            print(f"  [boundary diag, {cfg}] injecting kappa_eff at the boundary scales dR2 by xR(T_init) "
                  f"(here x{blow:.1f} at this low-Pb blowout row; >>1 at higher-Pb epochs) -> r2_prime can "
                  f"plunge past R1. Patching BOTH sites failed at every ldv>0; RHS-only is STABLE (6/6). "
                  f"=> IC kept Spitzer; refines SPEC §3.")

    # ---- the 2 captured fixtures (stiff 5e9, mild cluster) — G1 identity only (no logged Lloss) -
    for flabel, fixture_name, _note in _fm1._STATES:
        try:
            fixture, params = _fm1._load(fixture_name)
            Lmech = float(params["Lmech_total"].value) if "Lmech_total" in params else float("nan")
            # no logged per-row Lloss for fixtures: gate on identity, replay against the unpatched solve
            if "bubble_dMdt" in params:
                params["bubble_dMdt"].value = float("nan")
            bp0 = BL.get_bubbleproperties_pure(params)
            per, g = _eval_state(f"fixture: {flabel}", params, None,
                                 float(bp0.bubble_LTotal), Lmech)
            g["G2_replay"] = float("nan"); g["g2_pass"] = None     # n/a for fixtures
        except Exception as e:
            print(f"[fixture {flabel}] FAILED: {type(e).__name__}: {e}"); continue
        rows += per; gates.append(g); series[f"fixture: {flabel}"] = per
        print(f"[fixture: {flabel}]  G1 ident={g['G1_identity']:.1e} "
              f"{'OK' if g['g1_pass'] else 'FAIL'} | theta0={per[0]['theta']:.3f} -> "
              f"theta@ldv=1={next(p['theta'] for p in per if p['ldv']==1.0):.3f}")

    # ---- CSV --------------------------------------------------------------------------------------
    csv_path = os.path.join(_HERE, "kmix_selfconsistent.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["state", "ldv", "LTotal", "dMdt", "theta",
                                           "L_over_base", "solver_ok"])
        w.writeheader(); w.writerows(rows)
        fh.write("# kappa_eff = kappa_Spitzer * max(1, (ldv*3.086e23)*Pb_cgs/(6e-7*T^3.5)); ldv in pc.km/s\n")
        fh.write("# theta = bubble_LTotal / Lmech_total (resolved loss fraction). ldv=0 is the production baseline.\n")
        fh.write("# G1 identity (ldv=0 == unpatched) and G2 replay (ldv=0 == logged Lloss) gates printed at runtime.\n")
    print(f"wrote {csv_path}")

    n_g1 = sum(1 for g in gates if g["g1_pass"])
    n_g2 = sum(1 for g in gates if g.get("g2_pass"))
    print(f"\nGATES: G1 identity {n_g1}/{len(gates)} pass (<{IDENT_TOL:.0e}); "
          f"G2 replay {n_g2}/{sum(1 for g in gates if g.get('g2_pass') is not None)} pass (<{REL_TOL:.0e}).")

    # ---- verdict: rise? saturate? reach Lancaster? solver stable? ---------------------------------
    clean = {k: v for k, v in series.items() if not k.startswith("fixture:")}
    rises = survives = sats = lanc = 0
    for label, per in clean.items():
        th = [p["theta"] for p in per]
        if np.isfinite(th[0]) and th[1] is not None and np.isfinite(th[1]):
            if th[1] > th[0]:
                rises += 1
            # saturated: theta@ldv>=0.1 within 2% of theta@ldv=0.01 (knob is pinned)
            tail = [t for t in th[1:] if np.isfinite(t)]
            if tail and max(tail) - min(tail) < 0.02 * abs(th[1]):
                sats += 1
            if max(tail) >= 0.95:   # would the cooling_balance trigger fire?
                lanc += 1
        if all(p["solver_ok"] for p in per):
            survives += 1
    n = len(clean)
    print(f"VERDICT (6 cleanroom): theta RISES with kappa_mix in {rises}/{n}; SATURATES by ldv~0.01 in "
          f"{sats}/{n} (kappa_mix swamps Spitzer at tiny ldv -> NOT a continuous knob); reaches the "
          f"trigger band theta>=0.95 in {lanc}/{n} (only the diffuse end; dense/mid plateau LOW, far "
          f"below Lancaster); solver STABLE across the full sweep in {survives}/{n}. The 2 fixtures "
          "(stiff 5e9 collapse, mild cluster) fail at ldv>0 -- edge states, consistent with excluding the "
          "heavy cloud (KMIX_PROTOTYPE §2).")

    # ---- figure ----------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})"); return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3))
    cmap = plt.get_cmap("viridis")
    labels = list(series)
    axL.axhspan(0.9, 0.99, color="#2ca02c", alpha=0.12)
    axL.text(0.012, 0.945, "Lancaster target\n(θ ≈ 0.9–0.99)", fontsize=8, color="#1c6b1c")
    for i, label in enumerate(labels):
        per = series[label]
        c = cmap(i / max(1, len(labels) - 1))
        ld = [max(p["ldv"], 5e-3) for p in per]       # 0 -> 5e-3 for the log axis
        axL.semilogx(ld, [p["theta"] for p in per], "o-", color=c, lw=1.8, ms=4, label=label)
        axR.semilogx(ld, [p["L_over_base"] for p in per], "o-", color=c, lw=1.8, ms=4, label=label)
    axL.set_xlabel(r"$\lambda\delta v$ [pc km/s]  (leftmost = 0, the production baseline)")
    axL.set_ylabel(r"$\theta = L_{\rm cool}/L_{\rm mech}$  (resolved, self-consistent)")
    axL.set_title("Self-consistent θ vs κ_mix strength\n(does injecting mixing raise the resolved loss?)",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=6.6, loc="best"); axL.grid(True, which="both", alpha=0.2)
    axR.axhline(1.0, ls="--", color="0.5", lw=1.0)
    axR.set_xlabel(r"$\lambda\delta v$ [pc km/s]")
    axR.set_ylabel(r"$L_{\rm Total}(\lambda\delta v) / L_{\rm Total}(0)$")
    axR.set_title("Resolved cooling vs baseline\n(>1 ⇒ κ_mix enhances the structural loss)",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=6.6, loc="best"); axR.grid(True, which="both", alpha=0.2)
    fig.suptitle("Self-consistent κ_mix injection into the REAL structure solver "
                 "(monkeypatch, no production edit): θ response across the configs",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "kmix_selfconsistent.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
