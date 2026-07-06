#!/usr/bin/env python3
"""The QUEUED-BUT-NEVER-RUN second offline prototype of the KAPPA_EFF_SCOPING §6.2 redirect:
boost the interface COOLING SOURCE TERM inside the structure ODE (kappa untouched) and measure
Delta L_cool vs Delta dMdt -- "the new make-or-break".

WHY THIS KNOB (and how it differs from everything already tried)
----------------------------------------------------------------
The workstream has measured three structural options and one production fallback:
  - f_kappa on Spitzer C (Rung A):   cooling UP but evaporation UP too (dMdt x1.08-1.17 at f=2,
    KAPPA_EFF_SCOPING §6a) -- the WRONG-SIGN coupling (El-Badry Eq 47: mdot FALLS as theta rises),
    and it walks the solver into the condensation domain edge (KAPPA_FREEZE_MECHANISM).
  - kappa_mix floor (Rung B):        physical but BORN SATURATED (not a dial) and NaNs at the
    decisive early high-Pb epochs under hard-max (KMIX_SELFCONSISTENT §2b).
  - multiplier (production, f_mix=4): stable but NO back-reaction -- the structure never feels
    the loss (theta is emergent-then-scaled, dMdt untouched).
This harness tests the fourth corner: f_A multiplies the NET radiative source dudt ONLY in the
interface band (T < 10^5.5 K, the non-CIE regime) INSIDE _get_bubble_ODE, i.e. the 1-D projection
of Lancaster's fractal-area/turbulent-mixing enhancement (area factor on the interface emissivity;
El-Badry result vi: "mixing, not conduction, sets the cooling -- Spitzer mainly sets interior
T/evaporation"). The conduction operator, the Eq-44 Spitzer IC family, and the dMdt seed are all
UNTOUCHED, so the stiff machinery is never re-derived. Predictions this run tests:
  P1 (dial):      theta rises CONTINUOUSLY with f_A (no kappa_mix-style saturation -- the source
                  term is linear in f_A, it does not change the diffusion operator's order).
  P2 (sign):      the dMdt eigenvalue FALLS as f_A rises (radiated flux no longer drives
                  evaporation -- the El-Badry coupling, the thing f_kappa provably got backwards).
  P3 (stability): no early-epoch NaNs (the stiff prefactor Pb/(C T^{5/2}) is untouched).
  P4 (edge):      at large f_A, dMdt -> 0/negative = condensation onset (the McKee-Cowie
                  reversal); in production that is fix #1's no-root handoff, here it is recorded.

THE INJECTION (monkeypatch, no production edit; mirrors make_kmix_selfconsistent's strategy)
--------------------------------------------------------------------------------------------
_get_bubble_ODE is re-emitted VERBATIM except:  dudt -> f_A*dudt  when T < 10^5.5 K and f_A != 1
(at f_A=1 the branch is never entered -> literally the production float ops, G1 bit-identity).
The emergent loss is read as  L_eff = L1_bubble + f_A*(L2_conduction + L3_intermediate)  from the
returned dataclass -- the same band-limited area factor applied to the loss integrals, consistent
with the in-ODE source. (Note the production `multiplier` knob scales L1 too; the area argument
only justifies scaling the interface bands. theta here = L_eff/Lmech_total.)

CORRECTNESS GATES (HARD -- abort/flag, never fabricate):
  G1 IDENTITY: at f_A=1 the patched solve == the unpatched production solve to <1e-12.
  G2 REPLAY:   at f_A=1 the replayed bubble_LTotal == the logged bubble_Lloss to 1e-3
               (same gate as make_da_replay / make_kmix_selfconsistent).

COVERAGE: the 6 cleanroom configs via make_da_replay build_params/replay_row on the committed
cleanroom trajectories, ~N_ROWS rows per implicit-phase trajectory (the wrong-epoch lesson of
KMIX_SELFCONSISTENT §2b: never judge from the blowout row alone). NO sims.

REPRODUCE (from repo root; reads committed trajectories, ~8 min):
    python docs/dev/transition/pdv-trigger/data/make_fA_source_boost.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fA_source_boost.csv
    docs/dev/transition/pdv-trigger/data/fA_source_boost_summary.csv
    docs/dev/transition/pdv-trigger/fA_source_boost.png
"""
import csv
import logging
import os
import sys

import numpy as np

logging.disable(logging.CRITICAL)  # silence trinity's chatty solver logs

import trinity.bubble_structure.bubble_luminosity as BL
from trinity.cooling import net_coolingcurve

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import make_da_replay as DR        # noqa: E402  build_params(cfg), replay_row(params, row)

T_BAND = 10 ** 5.5                 # interface band = the non-CIE regime (same cut as the L2 zone)
FIRE = 0.95                        # cooling_balance trigger threshold
IDENT_TOL = 1e-12                  # G1 identity gate
REL_TOL = 1e-3                     # G2 replay gate (same as make_da_replay)
FA_LIST = [float(x) for x in os.environ.get("FA_LIST", "1,2,4,8,16").split(",")]
N_ROWS = int(os.environ.get("N_ROWS", "10"))
CONFIGS = os.environ.get("CONFIGS", "").split(",") if os.environ.get("CONFIGS") else None

_ORIG_ODE = BL._get_bubble_ODE


def _patched_ODE(fA):
    """_get_bubble_ODE with dudt -> fA*dudt in the interface band (T < 10^5.5 K) ONLY.
    Mirrors production bubble_luminosity.py:393-421; at fA=1 the boost branch is never entered,
    so the float ops are the production ops verbatim -> G1 bit-identity."""
    def ode(r_arr, initial_ODEs, params, Pb):
        v, T, dTdr = initial_ODEs
        if np.abs(T - 0) < 1e-5:
            raise BL.BubbleSolverError(
                f'temperature reached zero in bubble ODE RHS (T={T:.3e})')
        ndens = Pb / ((params['mu_convert'].value / params['mu_ion'].value)
                      * params['k_B'].value * T)
        phi = params['Qi'].value / (4 * np.pi * r_arr ** 2)
        dudt = net_coolingcurve.get_dudt(params['t_now'].value, ndens, T, phi, params)
        if fA != 1.0 and T < T_BAND:
            dudt = fA * dudt          # the interface area/emissivity factor (source term only)
        v_term = params['cool_alpha'].value * r_arr / params['t_now'].value
        dTdrr = (Pb / (params['cooling_boost_kappa'].value * params['C_thermal'].value
                       * T ** (5 / 2)) * (
            (params['cool_beta'].value + 2.5 * params['cool_delta'].value)
            / params['t_now'].value
            + 2.5 * (v - v_term) * dTdr / T - dudt / Pb
        ) - 2.5 * dTdr ** 2 / T - 2 * dTdr / r_arr)
        dvdr = ((params['cool_beta'].value + params['cool_delta'].value)
                / params['t_now'].value
                + (v - v_term) * dTdr / T - 2 * v / r_arr)
        return [dvdr, dTdr, dTdrr]
    return ode


def _solve(params, fA):
    """Full production solve with the fA source boost; return (L_eff, dMdt, L1, L2, L3, ok).
    L_eff = L1 + fA*(L2+L3): the same band factor applied to the loss integrals over the
    (back-reacted) profile, consistent with the in-ODE source."""
    BL._get_bubble_ODE = _patched_ODE(fA)
    try:
        if "bubble_dMdt" in params:
            params["bubble_dMdt"].value = float("nan")   # cold Eq-33 seed: deterministic per call
        bp = BL.get_bubbleproperties_pure(params)
        L_eff = float(bp.bubble_L1Bubble
                      + fA * (bp.bubble_L2Conduction + bp.bubble_L3Intermediate))
        return (L_eff, float(bp.bubble_dMdt), float(bp.bubble_L1Bubble),
                float(bp.bubble_L2Conduction), float(bp.bubble_L3Intermediate), True)
    except Exception:
        return (float("nan"),) * 5 + (False,)
    finally:
        BL._get_bubble_ODE = _ORIG_ODE


def _sample_rows(df, n_rows):
    """~n_rows rows with Pb>0 and finite logged Lloss, evenly spaced across the phase."""
    d = df[(df["Pb"] > 0) & np.isfinite(df["bubble_Lloss"])].reset_index(drop=True)
    if len(d) <= n_rows:
        return d
    idx = np.linspace(0, len(d) - 1, n_rows).round().astype(int)
    return d.iloc[sorted(set(idx.tolist()))].reset_index(drop=True)


def main():
    import pandas as pd

    configs = CONFIGS or DR.DS.CLEANROOM
    rows_out, summary, gates = [], [], []
    series = {}    # label -> dict(t=[], theta at each fA=[], dMdt at each fA=[])

    for cfg in configs:
        try:
            params = DR.build_params(cfg)
            df = pd.read_csv(f"docs/dev/transition/cleanroom/data/c0_{cfg}_h0.csv")
            sample = _sample_rows(df, N_ROWS)
            if len(sample) < 2:
                print(f"[{cfg}] <2 usable rows -- skip"); continue
        except Exception as e:
            print(f"[{cfg}] setup failed: {type(e).__name__}: {e}"); continue

        label = f"{cfg} (n={DR.DS.NCORE[cfg]:.0e})"
        ser = {"t": [], "theta": {fa: [] for fa in FA_LIST}, "dMdt": {fa: [] for fa in FA_LIST}}
        g1 = g2 = float("nan")
        for irow, (_, row) in enumerate(sample.iterrows()):
            DR.replay_row(params, row)
            Lmech = float(row["Lmech_total"])
            # G1/G2 on the first row only (one unpatched solve per config)
            if irow == 0:
                if "bubble_dMdt" in params:
                    params["bubble_dMdt"].value = float("nan")
                bp0 = BL.get_bubbleproperties_pure(params)      # unpatched
                L_unpatched = float(bp0.bubble_LTotal)
                L_id, _, l1, l2, l3, ok_id = _solve(params, 1.0)
                g1 = (abs(L_id - L_unpatched) / abs(L_unpatched)
                      if (ok_id and L_unpatched) else float("inf"))
                g2 = (abs(L_id - float(row["bubble_Lloss"])) / abs(float(row["bubble_Lloss"]))
                      if (ok_id and row["bubble_Lloss"]) else float("inf"))
            ser["t"].append(float(row["t_now"]))
            for fa in FA_LIST:
                L_eff, dMdt, l1, l2, l3, ok = _solve(params, fa)
                theta = (L_eff / Lmech) if (np.isfinite(L_eff) and Lmech) else float("nan")
                ser["theta"][fa].append(theta)
                ser["dMdt"][fa].append(dMdt)
                rows_out.append(dict(
                    config=cfg, nCore=DR.DS.NCORE[cfg], t_now=float(row["t_now"]),
                    R2=float(row["R2"]), fA=fa, Lmech=Lmech, L1=l1, L2=l2, L3=l3,
                    L_eff=L_eff, theta=theta, dMdt=dMdt, solver_ok=ok))
        gates.append(dict(config=cfg, G1_identity=g1, G2_replay=g2,
                          g1_pass=bool(np.isfinite(g1) and g1 < IDENT_TOL),
                          g2_pass=bool(np.isfinite(g2) and g2 < REL_TOL)))
        series[label] = ser

        # per-config summary at each fA
        t = np.asarray(ser["t"])
        base_dMdt = np.asarray(ser["dMdt"][1.0], float)
        for fa in FA_LIST:
            th = np.asarray(ser["theta"][fa], float)
            dm = np.asarray(ser["dMdt"][fa], float)
            fin = np.isfinite(th)
            with np.errstate(invalid="ignore", divide="ignore"):
                dm_ratio = dm / base_dMdt
            summary.append(dict(
                config=cfg, nCore=DR.DS.NCORE[cfg], fA=fa,
                theta_max=float(np.nanmax(th)) if fin.any() else float("nan"),
                theta_med=float(np.nanmedian(th)) if fin.any() else float("nan"),
                dMdt_ratio_med=(float(np.nanmedian(dm_ratio))
                                if np.isfinite(dm_ratio).any() else float("nan")),
                n_solved=int(fin.sum()), n_sampled=len(t),
                n_dMdt_neg=int(np.sum(np.isfinite(dm) & (dm <= 0))),
                fires=bool(fin.any() and np.nanmax(th) >= FIRE)))
        gl = gates[-1]
        srow = {s["fA"]: s for s in summary if s["config"] == cfg}
        print(f"[{label}] G1={gl['G1_identity']:.1e} {'OK' if gl['g1_pass'] else 'FAIL'} | "
              f"G2={gl['G2_replay']:.1e} {'OK' if gl['g2_pass'] else 'FAIL'} | "
              + " | ".join(f"fA={fa:g}: th_max={srow[fa]['theta_max']:.2f} "
                           f"dM~{srow[fa]['dMdt_ratio_med']:.2f} "
                           f"ok={srow[fa]['n_solved']}/{srow[fa]['n_sampled']}"
                           f"{' NEG' if srow[fa]['n_dMdt_neg'] else ''}"
                           for fa in FA_LIST))

    # ---- CSVs -----------------------------------------------------------------------------------
    with open(os.path.join(_HERE, "fA_source_boost.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "nCore", "t_now", "R2", "fA", "Lmech",
                                           "L1", "L2", "L3", "L_eff", "theta", "dMdt",
                                           "solver_ok"])
        w.writeheader(); w.writerows(rows_out)
        fh.write("# fA multiplies the net radiative source dudt in the interface band "
                 "(T<10^5.5 K) INSIDE _get_bubble_ODE; kappa/ICs untouched.\n")
        fh.write("# L_eff = L1 + fA*(L2+L3); theta = L_eff/Lmech_total. fA=1 = production "
                 "baseline (G1 bit-identity gated).\n")
    with open(os.path.join(_HERE, "fA_source_boost_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "nCore", "fA", "theta_max", "theta_med",
                                           "dMdt_ratio_med", "n_solved", "n_sampled",
                                           "n_dMdt_neg", "fires"])
        w.writeheader(); w.writerows(summary)
    print("wrote fA_source_boost{,_summary}.csv")

    n_g1 = sum(1 for g in gates if g["g1_pass"]); n_g2 = sum(1 for g in gates if g["g2_pass"])
    print(f"\nGATES: G1 identity {n_g1}/{len(gates)} (<{IDENT_TOL:.0e}); "
          f"G2 replay {n_g2}/{len(gates)} (<{REL_TOL:.0e}).")

    # ---- verdict on the four predictions ----------------------------------------------------------
    by_cfg = {}
    for s in summary:
        by_cfg.setdefault(s["config"], {})[s["fA"]] = s
    fa_hi = max(f for f in FA_LIST if f > 1) if any(f > 1 for f in FA_LIST) else None
    p1 = p2 = p3 = 0; fires_at = {}
    for cfg, d in by_cfg.items():
        fas = sorted(d)
        th = [d[f]["theta_max"] for f in fas]
        # P1 dial: theta_max strictly increases across the sweep (where finite)
        fin = [x for x in th if np.isfinite(x)]
        if len(fin) >= 2 and all(b > a for a, b in zip(fin, fin[1:])):
            p1 += 1
        # P2 sign: median dMdt ratio at the top fA < 1 (evaporation suppressed)
        if fa_hi and np.isfinite(d[fa_hi]["dMdt_ratio_med"]) and d[fa_hi]["dMdt_ratio_med"] < 1:
            p2 += 1
        # P3 stability: every sampled row solves at every fA
        if all(d[f]["n_solved"] == d[f]["n_sampled"] for f in fas):
            p3 += 1
        fires_at[cfg] = next((f for f in fas if d[f]["fires"]), None)
    n = len(by_cfg)
    print("\nVERDICT (the four predictions):")
    print(f"  P1 dial (theta_max rises monotonically with fA):        {p1}/{n}")
    print(f"  P2 El-Badry sign (dMdt FALLS at fA={fa_hi:g}):           {p2}/{n}")
    print(f"  P3 stability (all rows solve at all fA):                {p3}/{n}")
    print(f"  P4 condensation onsets (dMdt<=0 rows) per config: "
          + ", ".join(f"{c}:{sum(d[f]['n_dMdt_neg'] for f in sorted(d))}"
                      for c, d in by_cfg.items()))
    print(f"  fires (theta_max>=0.95) at fA: "
          + ", ".join(f"{c}:{fires_at[c] if fires_at[c] is not None else '-'}"
                      for c in by_cfg))

    # ---- figure -----------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})"); return
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(18, 5.2))
    labels = sorted(series, key=lambda L: float(L.split("n=")[1].rstrip(")")))
    cmap = plt.get_cmap("viridis")

    # (L) theta(t) at fA=4 (the production-adopted multiplier value) vs baseline
    fa_show = 4.0 if 4.0 in FA_LIST else FA_LIST[-1]
    axL.axhspan(0.9, 0.99, color="#2ca02c", alpha=0.12)
    axL.axhline(FIRE, ls="--", color="#d1495b", lw=1.1)
    for i, label in enumerate(labels):
        ser = series[label]; c = cmap(i / max(1, len(labels) - 1))
        axL.plot(ser["t"], ser["theta"][fa_show], "-", color=c, lw=2, label=label)
        axL.plot(ser["t"], ser["theta"][1.0], ":", color=c, lw=1.2)
    axL.set_xlabel("t [Myr] (implicit phase)")
    axL.set_ylabel(r"$\theta = L_{\rm eff}/L_{\rm mech}$")
    axL.set_title(f"$\\theta(t)$ at $f_A$={fa_show:g} (solid) vs baseline (dotted)",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=6.4); axL.grid(True, alpha=0.2)

    # (M) theta_max vs fA -- the dial
    for i, label in enumerate(labels):
        ser = series[label]; c = cmap(i / max(1, len(labels) - 1))
        th_max = [np.nanmax(ser["theta"][fa]) if np.isfinite(ser["theta"][fa]).any()
                  else np.nan for fa in FA_LIST]
        axM.semilogx(FA_LIST, th_max, "o-", color=c, lw=1.8, ms=4, label=label)
    axM.axhline(FIRE, ls="--", color="#d1495b", lw=1.1)
    axM.axhspan(0.9, 0.99, color="#2ca02c", alpha=0.12)
    axM.set_xlabel(r"$f_A$ (interface source boost)")
    axM.set_ylabel(r"$\theta_{\rm max}$ over sampled phase")
    axM.set_title("P1: is $f_A$ a continuous dial?\n(κ_mix saturated here; f_κ crashed)",
                  fontsize=10, fontweight="bold")
    axM.legend(fontsize=6.4); axM.grid(True, which="both", alpha=0.2)

    # (R) dMdt ratio vs fA -- the coupling direction
    for i, label in enumerate(labels):
        ser = series[label]; c = cmap(i / max(1, len(labels) - 1))
        base = np.asarray(ser["dMdt"][1.0], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratios = [float(np.nanmedian(np.asarray(ser["dMdt"][fa], float) / base))
                      for fa in FA_LIST]
        axR.semilogx(FA_LIST, ratios, "o-", color=c, lw=1.8, ms=4, label=label)
    axR.axhline(1.0, ls="--", color="0.5", lw=1.0)
    axR.set_xlabel(r"$f_A$")
    axR.set_ylabel(r"median $\dot M(f_A)\,/\,\dot M(1)$")
    axR.set_title("P2: does evaporation FALL as interface cooling rises?\n"
                  "(El-Badry Eq 47 sign; $f_\\kappa$ moved it the WRONG way)",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=6.4); axR.grid(True, which="both", alpha=0.2)

    fig.suptitle("Interface source-term boost $f_A$ inside the REAL structure solve "
                 "(monkeypatch, no production edit)", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(_PDV, "fA_source_boost.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
