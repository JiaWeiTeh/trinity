#!/usr/bin/env python3
"""REAL Damkohler-number replay + go/no-go re-screen for theta_target(Da).

WHAT THIS IS (and how it differs from make_da_screen.py)
--------------------------------------------------------
make_da_screen.py screened a PROXY: Da_shape = (R2/v2)*Pb, i.e. the real
Da = t_turb/t_cool,int evaluated AT A FIXED characteristic interface temperature
T_int (so Lambda(T_int) and the per-config T_int collapse into a swept constant C).
That proxy was NON-MONOTONIC in nCore (pl2_steep n=1e5 fell below large_diffuse
n=1e2) -> verdict NO-GO.

This script computes the REAL Da by REPLAYING trinity's own interface-cooling
calculation on the committed frozen per-step trajectories -- NO full sim re-run,
NO solver edit. For each (config, converged row) it rebuilds the exact param state
the bubble solver saw and re-invokes get_bubbleproperties_pure(), then reconstructs
the intermediate (interface) zone of _bubble_luminosity to extract:
    T_int       : emission-weighted interface temperature [K]
    n_int       : emission-weighted interface H-nuclei density [cm^-3]
    eps_int     : emission-weighted interface net volumetric cooling rate [erg/cm^3/s]
    t_cool,int  : (3/2) n_int k_B T_int / eps_int   [Myr]   (= (3/2)kB T_int/(n_int Lambda))
    t_turb      : R2 / v2                            [Myr]
    Da          : t_turb / t_cool,int               (dimensionless, REAL)

VALIDATION GATE (Step 1, HARD): the replay must reproduce the logged bubble_Lloss
per row (the full resolved cooling integral) to within REL_TOL. The interface zone
(L3) is checked to be bit-identical to the dataclass's bubble_L3Intermediate. If the
gate fails the script aborts -- it does not fabricate a Da.

UNITS: trinity carries code units (au): n [pc^-3], cooling rate dudt [au]. We convert
to cgs with cvt.ndens_au2cgs (pc^-3 -> cm^-3) and 1/cvt.dudt_cgs2au (au -> erg/cm^3/s)
ONLY at the t_cool definition; the bubble_Lloss gate proves the units are right.

STEP 2: re-screen the REAL Da with make_da_screen.py's screen logic (theta_da floor,
cb trigger, first_fire, fire_minus_blowout), sweeping C and theta_max, asking whether
ONE (C, theta_max) fires all 6 cleanroom configs sustained near blowout.

Deliverables (run from repo root):
  docs/dev/transition/pdv-trigger/data/da_replay.csv
  docs/dev/transition/pdv-trigger/da_replay.png

Run:
  python docs/dev/transition/pdv-trigger/data/make_da_replay.py
  python docs/dev/transition/pdv-trigger/data/make_da_replay.py --full   # all rows (slow)
"""
import logging
import sys

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
import matplotlib

matplotlib.use("Agg")  # headless; no usetex (mathtext only), matching make_theta_density_plot.py
import matplotlib.pyplot as plt

logging.disable(logging.CRITICAL)  # silence trinity's chatty solver logs

# --- trinity internals (the replay re-invokes the production bubble solver) ---------------------
from trinity._input.read_param import read_param
from trinity.sps import read_sps
from trinity.sps.update_feedback import get_current_sps_feedback
from trinity.cooling.non_CIE import read_cloudy as non_CIE
from trinity.phase0_init import get_InitCloudProp
import trinity.bubble_structure.bubble_luminosity as BL
import trinity._functions.operations as operations
import trinity._functions.unit_conversions as cvt

# --- reuse make_da_screen's screen machinery verbatim ------------------------------------------
sys.path.insert(0, "docs/dev/transition/pdv-trigger/data")
import make_da_screen as DS  # noqa: E402  (NCORE, CLEANROOM, C_GRID, THETA_MAX_GRID, first_fire, ...)

# Gate tolerance: replay bubble_LTotal vs logged bubble_Lloss (relative).
REL_TOL = 1e-3
KB_CGS = 1.380649e-16   # erg/K
MYR_S = 3.1556952e13    # s/Myr (Julian year)

# Per-config rows to replay (evenly spaced) unless --full. The blowout row + neighbours are always
# force-included so Da@blowout is exact. ~40 keeps the run ~6 min at ~1.4 s/solve.
N_SAMPLE = 40

# ===============================================================================================
# Interface-zone capture: wrap _bubble_luminosity so it ALSO records the intermediate (interface)
# zone diagnostics. This replicates lines ~588-787 of bubble_luminosity.py EXACTLY (verified:
# recomputed L3 == dataclass bubble_L3Intermediate bit-identically), reusing BL's own grid +
# solver + cooling tables, so it is a faithful read of trinity's interface, not a reimplementation
# of the physics.
# ===============================================================================================
_CAP = {}
_orig_bubble_luminosity = BL._bubble_luminosity


def _capturing_bubble_luminosity(params, R1, Pb, r2Prime, initial_conditions,
                                 bubble_r_Tb, bubble_dMdt):
    bp = _orig_bubble_luminosity(params, R1, Pb, r2Prime, initial_conditions,
                                 bubble_r_Tb, bubble_dMdt)
    try:
        _CAP.clear()
        _CAP.update(_interface_diagnostics(params, R1, Pb, r2Prime, initial_conditions))
        _CAP["L3_dataclass"] = float(bp.bubble_L3Intermediate)
    except Exception as e:  # never break the gate on a capture hiccup
        _CAP.clear()
        _CAP["error"] = repr(e)
    return bp


def _interface_diagnostics(params, R1, Pb, r2Prime, initial_conditions):
    """Reconstruct the intermediate (interface) zone exactly as _bubble_luminosity does."""
    mu_fac = params["mu_convert"].value / params["mu_ion"].value
    kB = params["k_B"].value
    _coolingswitch = 1e4
    _CIEswitch = 10 ** 5.5

    r_array = BL._create_radius_grid(R1, r2Prime)
    psoln, _ok, _info, _sol = BL._solve_bubble_structure(
        initial_conditions, r_array, params, Pb)
    v_array, T_array, dTdr_array = psoln[:, 0], psoln[:, 1], psoln[:, 2]
    n_array = Pb / (mu_fac * kB * T_array)

    index_CIE_switch = operations.find_nearest_higher(T_array, _CIEswitch)
    index_cooling_switch = operations.find_nearest_higher(T_array, _coolingswitch)
    cooling_CIE = params["cStruc_cooling_CIE_interpolation"].value

    if index_cooling_switch != index_CIE_switch:
        _xtra = 20
        r_interp = r_array[:index_CIE_switch + _xtra]
        fdTdr_interp = scipy.interpolate.interp1d(
            r_interp, dTdr_array[:index_CIE_switch + _xtra], kind="linear")
        fT_interp = scipy.interpolate.interp1d(
            r_interp, T_array[:index_CIE_switch + _xtra] - _CIEswitch, kind="cubic")
        fv_interp = scipy.interpolate.interp1d(
            r_interp, v_array[:index_CIE_switch + _xtra], kind="linear")
        r_CIEswitch = scipy.optimize.brentq(
            fT_interp, np.min(r_interp), np.max(r_interp), xtol=1e-8)
        n_CIEswitch = Pb / (mu_fac * kB * _CIEswitch)
        dTdr_CIEswitch = fdTdr_interp(r_CIEswitch)
        v_CIEswitch = fv_interp(r_CIEswitch)
        T_array = np.insert(T_array, index_CIE_switch, _CIEswitch)
        r_array = np.insert(r_array, index_CIE_switch, r_CIEswitch)
        n_array = np.insert(n_array, index_CIE_switch, n_CIEswitch)
        dTdr_array = np.insert(dTdr_array, index_CIE_switch, dTdr_CIEswitch)
        v_array = np.insert(v_array, index_CIE_switch, v_CIEswitch)

    dTdr_bubble = dTdr_array[index_CIE_switch:]
    dTdR_coolingswitch = dTdr_bubble[0]
    if index_cooling_switch != index_CIE_switch:
        r_conduction = np.linspace(
            r_array[0], r_array[index_CIE_switch], BL._CONDUCTION_NPTS)
        _cond = _sol.sol(r_conduction)
        T_cond, dTdr_cond = _cond[1], _cond[2]
        mask = T_cond < _CIEswitch
        dTdr_cond = dTdr_cond[mask]
        dTdR_coolingswitch = dTdr_cond[0] if len(dTdr_cond) > 0 else dTdr_bubble[0]

    R2_coolingswitch = ((_coolingswitch - T_array[index_cooling_switch])
                        / dTdR_coolingswitch + r_array[index_cooling_switch])
    fT_interp_interm = scipy.interpolate.interp1d(
        np.array([r_array[index_cooling_switch], R2_coolingswitch]),
        np.array([T_array[index_cooling_switch], _coolingswitch]), kind="linear")

    r_interm = np.linspace(r_array[index_cooling_switch], R2_coolingswitch,
                           num=1000, endpoint=True)
    T_interm = fT_interp_interm(r_interm)
    n_interm = Pb / (mu_fac * kB * T_interm)
    phi_interm = params["Qi"].value / (4 * np.pi * r_interm ** 2)

    # per-cell |net volumetric cooling rate| [au] (the L3 integrand without 4 pi r^2)
    emiss = np.zeros_like(r_interm)
    L3 = 0.0
    for regime in ["non-CIE", "CIE"]:
        mask = T_interm < _CIEswitch if regime == "non-CIE" else T_interm >= _CIEswitch
        if not np.any(mask):
            continue
        if regime == "non-CIE":
            cN = params["cStruc_cooling_nonCIE"].value
            hN = params["cStruc_heating_nonCIE"].value
            cool_int = 10 ** cN.interp(np.transpose(np.log10(
                [n_interm[mask] / cvt.ndens_cgs2au, T_interm[mask],
                 phi_interm[mask] / cvt.phi_cgs2au])))
            heat_int = 10 ** hN.interp(np.transpose(np.log10(
                [n_interm[mask] / cvt.ndens_cgs2au, T_interm[mask],
                 phi_interm[mask] / cvt.phi_cgs2au])))
            dudt_int = (heat_int - cool_int) * cvt.dudt_cgs2au
            integ = dudt_int * 4 * np.pi * r_interm[mask] ** 2
            emiss[mask] = dudt_int
        else:
            Lambda_int = 10 ** (cooling_CIE(np.log10(T_interm[mask]))) * cvt.Lambda_cgs2au
            integ = (params["chi_e"].value * n_interm[mask] ** 2 * Lambda_int
                     * 4 * np.pi * r_interm[mask] ** 2)
            emiss[mask] = params["chi_e"].value * n_interm[mask] ** 2 * Lambda_int
        L3 += np.abs(BL._trapezoid(integ, x=r_interm[mask]))

    w = np.abs(emiss)
    sw = float(np.sum(w))
    if sw <= 0 or not np.isfinite(sw):
        raise ValueError("interface emissivity weight is zero/non-finite")
    T_int = float(np.sum(w * T_interm) / sw)
    n_int_au = float(np.sum(w * n_interm) / sw)       # emission-weighted, code units (pc^-3)
    eps_int_au = float(np.sum(w * np.abs(emiss)) / sw)  # emission-weighted |net rate|, au

    return dict(
        L3_recomputed=float(L3),
        T_int=T_int,
        T_interm_min=float(T_interm.min()),
        T_interm_max=float(T_interm.max()),
        n_int_cgs=n_int_au * cvt.ndens_au2cgs,                 # cm^-3
        eps_int_cgs=eps_int_au / cvt.dudt_cgs2au,              # erg/cm^3/s
    )


# ===============================================================================================
# Per-config param rebuild + per-row replay
# ===============================================================================================
def build_params(cfg):
    """Rebuild the param state for `cfg` exactly as main.start_expansion does (phase0)."""
    params = read_param(f"docs/dev/transition/cleanroom/configs/{cfg}.param")
    get_InitCloudProp.get_InitCloudProp(params)
    params["sps_data"].value = read_sps.read_sps(
        params["mCluster"] / params["sps_refmass"], params)
    params["sps_f"].value = read_sps.get_interpolation(params["sps_data"].value)
    logT, logLambda = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    params["cStruc_cooling_CIE_logLambda"].value = logLambda
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logLambda, kind="linear")
    return params


def replay_row(params, row):
    """Set per-row solver state, re-invoke the bubble solver, return (bp, capture, fb)."""
    t = float(row["t_now"])
    fb = get_current_sps_feedback(t, params)  # SPS at t -> Qi, v_mech_total, Lmech_total
    params["t_now"].value = t
    params["R2"].value = float(row["R2"])
    params["v2"].value = float(row["v2"])
    params["Eb"].value = float(row["Eb"])
    params["T0"].value = float(row["T0"])
    params["Lmech_total"].value = float(row["Lmech_total"])
    params["v_mech_total"].value = float(fb.v_mech_total)
    params["Qi"].value = float(fb.Qi)
    params["cool_beta"].value = float(row["cool_beta"])
    params["cool_delta"].value = float(row["cool_delta"])
    params["cool_alpha"].value = t / float(row["R2"]) * float(row["v2"])
    params["bubble_dMdt"].value = np.nan  # force the dMdt fsolve to re-solve from scratch
    # time-dependent non-CIE cooling cube at this age (interface zone uses it)
    cN, hN, nN = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cN
    params["cStruc_heating_nonCIE"].value = hN
    params["cStruc_net_nonCIE_interpolation"].value = nN
    bp = BL.get_bubbleproperties_pure(params)
    return bp, dict(_CAP), fb


def select_rows(d, blow_i, n_sample, full):
    """Indices to replay: all (full) or ~n_sample evenly spaced + blowout row + neighbours."""
    n = len(d)
    if full or n <= n_sample:
        return list(range(n))
    idx = set(np.linspace(0, n - 1, n_sample).round().astype(int).tolist())
    if blow_i is not None:
        for j in range(max(0, blow_i - 2), min(n, blow_i + 3)):
            idx.add(j)
    return sorted(idx)


# ===============================================================================================
# Main: replay -> gate -> per-row Da -> re-screen -> CSV + PNG
# ===============================================================================================
def main():
    full = "--full" in sys.argv
    BL._bubble_luminosity = _capturing_bubble_luminosity  # install the interface capture

    rcloud = DS.load_rcloud()
    series = {}        # cfg -> per-row DataFrame with Da etc.
    gate_stats = []    # (cfg, n_rows, max_reldiff_Lloss, max_reldiff_L3)
    blow_info = {}     # cfg -> dict at blowout

    for cfg in DS.CLEANROOM:
        df = pd.read_csv(f"docs/dev/transition/cleanroom/data/c0_{cfg}_h0.csv")
        df = df[df["betadelta_converged"] == True]  # noqa: E712
        df = df[df["Eb"] > 0].reset_index(drop=True)
        rc = rcloud.get(cfg)
        blow_i = None
        if rc is not None and (df["R2"] >= rc).any():
            blow_i = int(np.argmax((df["R2"] >= rc).to_numpy()))

        params = build_params(cfg)
        rows_out = []
        max_rd_lloss = 0.0
        max_rd_l3 = 0.0
        sel = select_rows(df, blow_i, N_SAMPLE, full)
        print(f"[{cfg}] replaying {len(sel)} rows "
              f"(blowout row idx={blow_i})...", flush=True)
        for i in sel:
            row = df.iloc[i]
            try:
                bp, cap, fb = replay_row(params, row)
            except Exception as e:
                print(f"  row {i} (t={row['t_now']:.4f}) solve FAILED: {e}", flush=True)
                continue
            if "error" in cap or "T_int" not in cap:
                print(f"  row {i}: interface capture failed: {cap.get('error')}", flush=True)
                continue
            lloss_csv = float(row["bubble_Lloss"])
            rd_lloss = abs(bp.bubble_LTotal - lloss_csv) / abs(lloss_csv) if lloss_csv else np.nan
            rd_l3 = (abs(cap["L3_recomputed"] - cap["L3_dataclass"])
                     / abs(cap["L3_dataclass"]) if cap["L3_dataclass"] else 0.0)
            if np.isfinite(rd_lloss):
                max_rd_lloss = max(max_rd_lloss, rd_lloss)
            max_rd_l3 = max(max_rd_l3, rd_l3)

            t_cool_s = (1.5 * cap["n_int_cgs"] * KB_CGS * cap["T_int"]
                        / cap["eps_int_cgs"])
            t_cool_Myr = t_cool_s / MYR_S
            t_turb_Myr = float(row["R2"]) / float(row["v2"])  # pc / (pc/Myr) = Myr
            Da = t_turb_Myr / t_cool_Myr

            rows_out.append(dict(
                idx=i, t=float(row["t_now"]), R2=float(row["R2"]), v2=float(row["v2"]),
                Pb=float(row["Pb"]), Eb=float(row["Eb"]),
                Lmech=float(row["Lmech_total"]), Lcool=float(row["bubble_Lloss"]),
                Lloss_replay=float(bp.bubble_LTotal), reldiff_Lloss=rd_lloss,
                reldiff_L3=rd_l3,
                T_int=cap["T_int"], T_interm_min=cap["T_interm_min"],
                T_interm_max=cap["T_interm_max"], n_int_cgs=cap["n_int_cgs"],
                eps_int_cgs=cap["eps_int_cgs"],
                t_cool_int_Myr=t_cool_Myr, t_turb_Myr=t_turb_Myr, Da=Da,
                is_blowout=(blow_i is not None and i == blow_i),
            ))
        fr = pd.DataFrame(rows_out)
        series[cfg] = fr
        gate_stats.append((cfg, len(fr), max_rd_lloss, max_rd_l3))
        print(f"  -> {len(fr)} rows; max reldiff bubble_Lloss={max_rd_lloss:.2e}, "
              f"max reldiff L3={max_rd_l3:.2e}", flush=True)

    # -------- Step 1 GATE verdict -----------------------------------------------------------
    worst_lloss = max(g[2] for g in gate_stats)
    worst_l3 = max(g[3] for g in gate_stats)
    gate_pass = worst_lloss < REL_TOL
    print("\n" + "=" * 78)
    print(f"STEP 1 GATE: worst bubble_Lloss reldiff = {worst_lloss:.3e} "
          f"(tol {REL_TOL:.0e})  ->  {'PASS' if gate_pass else 'FAIL'}")
    print(f"             worst interface-L3 reldiff  = {worst_l3:.3e} "
          f"(should be ~0; bit-identical reconstruction)")
    print("=" * 78)
    if not gate_pass:
        print("GATE FAILED -- not computing Da. See per-config stats above.")
        return

    # -------- per-config Da@blowout + screen-ready frames -----------------------------------
    for cfg in DS.CLEANROOM:
        fr = series[cfg]
        bi = fr.index[fr["is_blowout"]].tolist()
        bi = bi[0] if bi else None
        b = fr.loc[bi] if bi is not None else None
        blow_info[cfg] = dict(
            Da_blow=(float(b["Da"]) if b is not None else np.nan),
            tcool_blow=(float(b["t_cool_int_Myr"]) if b is not None else np.nan),
            tturb_blow=(float(b["t_turb_Myr"]) if b is not None else np.nan),
            Tint_blow=(float(b["T_int"]) if b is not None else np.nan),
            nint_blow=(float(b["n_int_cgs"]) if b is not None else np.nan),
            t_blow=(float(b["t"]) if b is not None else np.nan),
            duration=float(fr["t"].iloc[-1] - fr["t"].iloc[0]),
        )

    # -------- Step 2: re-screen on the REAL Da ----------------------------------------------
    # Build SERIES-like frames for the screen: reuse DS.screen_one's mechanics by populating
    # DS.SERIES with frames carrying a "Da_shape" column = REAL Da (so theta_da uses real Da),
    # plus Lcool/Lleak/Lmech/t and a blowout index.
    DS.SERIES.clear()
    for cfg in DS.CLEANROOM:
        fr = series[cfg].copy()
        bi = fr.index[fr["is_blowout"]].tolist()
        bi = bi[0] if bi else None
        screen_frame = pd.DataFrame({
            "t": fr["t"], "R2": fr["R2"], "v2": fr["v2"], "Pb": fr["Pb"],
            "Lmech": fr["Lmech"], "Lcool": fr["Lcool"],
            "Lleak": pd.Series(0.0, index=fr.index), "Eb": fr["Eb"],
            "Da_shape": fr["Da"],  # REAL Da feeds theta_da
            "lossfrac": (fr["Lcool"]) / fr["Lmech"],
        }).reset_index(drop=True)
        bi2 = None
        if bi is not None:
            bi2 = int(fr.index.get_loc(bi))
        DS.SERIES[cfg] = dict(
            frame=screen_frame, rcloud=rcloud.get(cfg), bi=bi2,
            t_blow=blow_info[cfg]["t_blow"],
            R2_blow=(float(fr.loc[bi, "R2"]) if bi is not None else np.nan),
            Da_shape_blow=blow_info[cfg]["Da_blow"],
            lossfrac_blow=(float(screen_frame["lossfrac"].iloc[bi2])
                           if bi2 is not None else np.nan),
            Da_bulk_blow=np.nan,
            duration=blow_info[cfg]["duration"],
        )

    da_blows = {c: DS.SERIES[c]["Da_shape_blow"] for c in DS.CLEANROOM}
    da_norm = float(np.median([v for v in da_blows.values() if np.isfinite(v)]))
    key, n_valid, n_fire, C_best, tmax_best, per = DS.best_setting(da_norm)

    # -------- CSV ---------------------------------------------------------------------------
    out_rows = []
    for cfg in DS.CLEANROOM:
        binfo = blow_info[cfg]
        p = per[cfg]
        out_rows.append(dict(
            config=cfg, nCore=DS.NCORE[cfg],
            t_blowout=round(binfo["t_blow"], 6) if np.isfinite(binfo["t_blow"]) else np.nan,
            duration=round(binfo["duration"], 6),
            Da_real_blow=binfo["Da_blow"],
            Da_real_blow_norm=binfo["Da_blow"] / da_norm if np.isfinite(binfo["Da_blow"]) else np.nan,
            t_cool_int_blow_Myr=binfo["tcool_blow"],
            t_turb_blow_Myr=binfo["tturb_blow"],
            T_int_blow_K=binfo["Tint_blow"],
            n_int_blow_cgs=binfo["nint_blow"],
            lossfrac_blow=round(DS.SERIES[cfg]["lossfrac_blow"], 4)
            if np.isfinite(DS.SERIES[cfg]["lossfrac_blow"]) else np.nan,
            C_best=C_best, theta_max_best=tmax_best,
            theta_target_at_blow=round(p["theta_at_blow"], 4)
            if np.isfinite(p["theta_at_blow"]) else np.nan,
            da_fires=p["fires"], da_sustained=p["sustained"], da_birth=p["birth"],
            da_end_blip=p["end_blip"], da_fire_minus_blowout=p["fire_minus_blowout"],
            da_fmb_over_duration=(round(p["fire_minus_blowout"] / binfo["duration"], 4)
                                  if (not np.isnan(p["fire_minus_blowout"])
                                      and binfo["duration"]) else np.nan),
        ))
    out = pd.DataFrame(out_rows)
    dst_csv = "docs/dev/transition/pdv-trigger/data/da_replay.csv"
    out.to_csv(dst_csv, index=False)

    # -------- density-law fits --------------------------------------------------------------
    order = sorted(DS.CLEANROOM, key=lambda c: DS.NCORE[c])
    ns = np.array([DS.NCORE[c] for c in order], float)
    da_real = np.array([blow_info[c]["Da_blow"] for c in order], float)
    # theta_target(Da)@blowout = theta_max * Da/(1+Da); density law test theta/(1-theta) ~ n^p
    th_da = tmax_best * da_real / (1.0 + da_real)
    ratio_da = th_da / (1.0 - th_da)
    ok = np.isfinite(np.log10(ratio_da)) & np.isfinite(da_real) & (da_real > 0)
    p_da = np.polyfit(np.log10(ns[ok]), np.log10(ratio_da[ok]), 1)[0] if ok.sum() >= 2 else np.nan
    # also: is Da itself monotonic in nCore?
    da_by_n = da_real[np.argsort(ns)]
    monotone = bool(np.all(np.diff(da_by_n[np.isfinite(da_by_n)]) > 0))

    # -------- figure (Agg, mathtext, scatter for independent configs) -----------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(DS.CLEANROOM)))
    cc = dict(zip(order, colors))
    for cfg in order:
        fr = series[cfg]
        axL.plot(fr["t"], fr["Da"], color=cc[cfg], lw=1.5,
                 label=f"{cfg} (n={DS.NCORE[cfg]:.0e})", marker="o", ms=2.5)
        b = fr[fr["is_blowout"]]
        if len(b):
            axL.scatter(b["t"], b["Da"], color=cc[cfg], s=120, edgecolor="k",
                        linewidth=0.9, zorder=6)
    if np.isfinite(da_norm):
        axL.axhline(da_norm, color="0.4", ls=":", lw=1.2, label="median real Da@blowout")
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel("t  [Myr]")
    axL.set_ylabel(r"real Da $= t_{turb}/t_{cool,int}$")
    axL.set_title("REAL Da trajectories (markers = blowout)\n"
                  "replayed from trinity interface cooling")
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(fontsize=7.0, loc="best")

    # RIGHT: fire alignment under best (C, theta_max); scatter only (independent configs)
    xs = np.arange(len(order))
    da_fmb = [per[c]["fire_minus_blowout"] for c in order]
    da_dur = [DS.SERIES[c]["duration"] for c in order]
    da_norm_fmb = [(f / d if (not np.isnan(f) and d) else np.nan)
                   for f, d in zip(da_fmb, da_dur)]
    da_valid = [DS.valid_fire(per[c]) for c in order]
    axR.axhline(0.0, color="crimson", ls="--", lw=1.5, label="blowout (target)")
    axR.scatter(xs, da_norm_fmb, s=140, color="tab:blue", edgecolor="k", linewidth=0.7,
                zorder=5, label=f"real-Da target best (C={C_best:.3g}, "
                                f"theta_max={tmax_best})")
    for i, okv in enumerate(da_valid):
        if not okv and not np.isnan(da_norm_fmb[i]):
            axR.scatter([xs[i]], [da_norm_fmb[i]], s=280, facecolors="none",
                        edgecolors="red", linewidth=1.8, zorder=4)
    axR.set_xticks(xs)
    axR.set_xticklabels([f"{c}\nn={DS.NCORE[c]:.0e}" for c in order],
                        fontsize=7.0, rotation=20, ha="right")
    axR.set_ylabel("(t_fire - t_blowout) / run duration")
    axR.set_title("Fire alignment vs blowout (0 = at blowout)\n"
                  "red ring = fire NOT sustained / birth / end-blip")
    axR.grid(True, axis="y", alpha=0.3)
    axR.legend(fontsize=8.0, loc="best")

    GATE = 0.15
    go = (n_valid == len(DS.CLEANROOM)) and all(
        DS.valid_fire(per[c]) and abs(per[c]["fire_minus_blowout"])
        < GATE * DS.SERIES[c]["duration"] for c in DS.CLEANROOM)
    verdict = "GO" if go else "NO-GO"
    fig.suptitle(
        f"REAL Da theta_target(Da) go/no-go re-screen  ->  {verdict}   |   "
        f"best single knob C={C_best:.3g}, theta_max={tmax_best}, "
        f"valid sustained fires={n_valid}/6 (any-fire {n_fire}/6)   |   "
        f"Da monotone in nCore: {monotone}", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    dst_png = "docs/dev/transition/pdv-trigger/da_replay.png"
    fig.savefig(dst_png, dpi=150)

    # -------- console report ----------------------------------------------------------------
    print(f"\nwrote {dst_csv}  ({len(out)} rows)")
    print(f"wrote {dst_png}")
    print(f"\nreal Da normalization (median Da@blowout) = {da_norm:.4e}")
    finite_da = [v for v in da_blows.values() if np.isfinite(v)]
    if finite_da:
        print(f"real Da@blowout spread max/min = {max(finite_da) / min(finite_da):.2f}x")
    print(f"BEST single knob: C={C_best:.4g}, theta_max={tmax_best} "
          f"(valid {n_valid}/6, any-fire {n_fire}/6)\n")
    hdr = (f"{'config':22s} {'nCore':>7s} {'Da@blow':>11s} {'tcool[Myr]':>11s} "
           f"{'T_int[K]':>10s} {'th(Da)@blow':>11s} {'da_fmb/dur':>10s} {'valid':>5s}")
    print(hdr)
    for cfg in order:
        b = blow_info[cfg]
        p = per[cfg]
        dur = b["duration"]
        fmb = p["fire_minus_blowout"]
        th = tmax_best * b["Da_blow"] / (1 + b["Da_blow"]) if np.isfinite(b["Da_blow"]) else np.nan
        print(f"{cfg:22s} {DS.NCORE[cfg]:7.0e} {b['Da_blow']:11.3f} {b['tcool_blow']:11.3e} "
              f"{b['Tint_blow']:10.1f} {th:11.4f} "
              f"{(fmb / dur if (not np.isnan(fmb) and dur) else float('nan')):10.3f} "
              f"{str(DS.valid_fire(p)):>5s}")

    print(f"\nreal Da@blowout MONOTONE in nCore (ascending): {monotone}")
    print("  (the proxy Da_shape was NON-monotone -- pl2_steep below large_diffuse)")
    print("  Da@blowout ordered by nCore:")
    for cfg in order:
        print(f"    n={DS.NCORE[cfg]:.0e}  {cfg:22s}  Da={blow_info[cfg]['Da_blow']:.3f}")
    print(f"\nDensity law: theta_target(Da)@blowout/(1-theta) ~ n^p  -> p_Da = {p_da:.3f}")
    print("  (El-Badry sqrt(n) => p~0.5 ; linear-n Da ansatz => p~1.0). 6 points, SFE confounded.")

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}  (valid sustained fires {n_valid}/6 at best single knob)")
    if go:
        print("  Real Da separates the grid: a single (C, theta_max) fires all 6 configs")
        print("  sustained near blowout -> green-light solver implementation of theta_target(Da).")
    else:
        print("  Even the REAL Da does not line the configs up at blowout under one knob.")
        print(f"  real Da@blowout monotone in nCore: {monotone}; "
              f"spread {max(finite_da) / min(finite_da):.1f}x." if finite_da else "")
    print("=" * 78)
    return out


if __name__ == "__main__":
    main()
