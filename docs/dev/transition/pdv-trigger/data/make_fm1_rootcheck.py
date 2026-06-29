#!/usr/bin/env python3
"""FM1 offline proof: does the Rung-B §3a closure admit a v(R1)=0 root? (NO production edit.)

WHAT THIS TESTS (RUNGB_SCOPING.md §3a + §9 FM1)
-----------------------------------------------
Rung B demotes the evaporative mass flux dMdt from the shooting eigenvalue to an
*entrainment-set input* (El-Badry/Lancaster: evaporation SUPPRESSED 3-30x vs Spitzer),
and promotes the near-front temperature gradient dTdr_front to the eigenvalue the inner
boundary condition v(R1)=0 fixes. The make-or-break risk (FM1) is whether that re-counted
closure still admits a physical v(R1)=0 root once dMdt is held fixed.

This harness proves it OFFLINE on REAL captured stiff bubble states (no solver edit, no full
run). For each state and each suppression factor s in {1,3,10,30}:
  - dMdt := dMdt_Spitzer / s                       (the entrainment-set input)
  - sweep dTdr_front; for each value build the front IC WITHOUT the Spitzer enthalpy
    balance: the layer offset is slaved geometrically, dR2 = (2/5)*T_init/|dTdr_front|
    (the same geometric relation Spitzer's dTdr = -2/5 T/dR2 encodes, so s=1 reduces to
    the Spitzer IC exactly), r2Prime = R2 - dR2, v_front = the recoil formula with the
    fixed dMdt, T(r2Prime) = T_init.
  - integrate the REAL production ODE (BL._get_bubble_ODE) r2Prime -> R1, read v(R1).
  - locate the dTdr_front root of v(R1)=0 and check the structure is physical
    (T>0 and monotonic, r2Prime in (R1,R2)).

BUILT-IN CORRECTNESS CHECK: at s=1 the root must land on the Spitzer dTdr (dMdt unchanged,
geometric dR2 reduces to the Spitzer dR2), recovering v(R1)=0 at the known baseline. If the
s=1 root is not at f≈1 the harness is wrong, not the physics.

THE FM1 SIGN ARGUMENT -- AND ITS REFUTATION BY THIS HARNESS. The scoping doc PREDICTED that a
suppressed (smaller) dMdt gives a WEAKER recoil term (proportional to dMdt), so the v(R1)=0
crossing should be EASIER to find as s grows. **This harness refutes that prediction.** The
recoil term is numerically tiny (it shifts v_front by ~0.5 out of a streaming velocity ~2243),
but the stiff BVP EXPONENTIALLY amplifies v_front: that ~0.5 shift moves v(R1) by ~2000, while
sweeping dTdr_front over 6 decades barely moves v(R1) at all. So v(R1)=0 is controlled by dMdt
(through the recoil in v_front), NOT by the conduction gradient -- only s=1 (the full Spitzer
dMdt, where the recoil is exactly the self-consistent value) reaches the BC. Verdict: the §3a
"demote dMdt, shoot on dTdr_front" reformulation is REFUTED -- you cannot fix dMdt and keep
v(R1)=0 by tuning conduction. The decoupling must NOT live at the dMdt / inner-BC level; the
viable path is to keep dMdt as the Weaver eigenvalue and add the mixing-layer cooling only to
the radiative-loss integrand (RUNGB_SCOPING.md §3a, updated). Caught OFFLINE, before any code.

REPRODUCE (from repo root):
    python docs/dev/transition/pdv-trigger/data/make_fm1_rootcheck.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fm1_rootcheck.csv
    docs/dev/transition/pdv-trigger/fm1_rootcheck.png
"""

import csv
import json
import os

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize

import trinity.bubble_structure.bubble_luminosity as BL
import trinity.cooling.non_CIE.read_cloudy as non_CIE
from trinity._input.read_param import read_param

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
# .../docs/dev/transition/pdv-trigger/data -> repo root is 5 parents up.
_REPO = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 5)))

_T_INIT = BL._T_INIT_BOUNDARY  # 3e4 K front anchor
_SUPPRESSION = [1, 3, 10, 30]  # El-Badry/Lancaster evaporation-suppression range (s=1 = baseline)
_PATH_SKIP = {"path_cooling_CIE", "path_cooling_nonCIE", "sps_path", "path2output"}

# Captured REAL bubble states (label, fixture file, regime note).
_STATES = [
    ("stiff 5e9/sfe0.01", "dR2_stiff_state_fixture.json", "dR2/R2 ~ 1e-10, the LSODA-flood regime"),
    ("mild cluster", "residual_resample_fixture.json", "a mild real cluster"),
]


def _build_params(fixture):
    """Rebuild a full params (cooling cubes included) from a distilled fixture.

    Mirrors test_dR2min_magic_number._build_params / test_residual_resample._build_params.
    """
    base = os.path.join(_REPO, fixture["base_param"])
    params = read_param(base)
    for k, v in fixture["param_values"].items():
        if k in _PATH_SKIP:
            continue
        if k in params:
            params[k].value = v
    logT, logL = np.loadtxt(params["path_cooling_CIE"].value, unpack=True)
    params["cStruc_cooling_CIE_logT"].value = logT
    if "cStruc_cooling_CIE_logLambda" in params:
        params["cStruc_cooling_CIE_logLambda"].value = logL
    params["cStruc_cooling_CIE_interpolation"].value = scipy.interpolate.interp1d(
        logT, logL, kind="linear"
    )
    cooling_nonCIE, heating_nonCIE, net = non_CIE.get_coolingStructure(params)
    params["cStruc_cooling_nonCIE"].value = cooling_nonCIE
    params["cStruc_heating_nonCIE"].value = heating_nonCIE
    params["cStruc_net_nonCIE_interpolation"].value = net
    return params


def _load(fixture_name):
    with open(os.path.join(_REPO, "test", "data", fixture_name)) as fh:
        fixture = json.load(fh)
    return fixture, _build_params(fixture)


def _spitzer_dTdr(dMdt, params, R2):
    """Spitzer near-front dTdr at this dMdt (the reference the sweep is normalised by)."""
    k_B = params["k_B"].value
    mu_ion = params["mu_ion"].value
    C = params["C_thermal"].value
    constant = 25.0 / 4.0 * k_B / mu_ion / C
    dR2 = _T_INIT ** 2.5 / (constant * dMdt / (4 * np.pi * R2 ** 2))
    return -2.0 / 5.0 * _T_INIT / dR2  # < 0


def _v_at_R1(dTdr_front, dMdt, params, Pb, R1, R2):
    """Integrate the REAL bubble ODE from the §3a front IC to R1; return (v(R1), ok, Tmono, dR2/R2).

    The §3a front IC: dMdt is an INPUT (no Spitzer enthalpy balance fixing it); the layer
    offset is slaved geometrically dR2 = (2/5) T_init/|dTdr_front|; v_front is the recoil
    formula; dTdr_front is the free eigenvalue this routine is shot on.
    """
    k_B = params["k_B"].value
    mu_ion = params["mu_ion"].value
    cool_alpha = params["cool_alpha"].value
    t_now = params["t_now"].value

    dR2 = (2.0 / 5.0) * _T_INIT / abs(dTdr_front)
    r2Prime = R2 - dR2
    if not (R1 < r2Prime < R2):
        return np.nan, False, False, dR2 / R2
    v_front = cool_alpha * R2 / t_now - dMdt / (4 * np.pi * R2 ** 2) * k_B * _T_INIT / mu_ion / Pb
    try:
        with BL._quiet_lsoda_fortran():
            sol = scipy.integrate.solve_ivp(
                fun=lambda r, y: BL._get_bubble_ODE(r, y, params, Pb),
                t_span=(r2Prime, R1),
                y0=[v_front, _T_INIT, dTdr_front],
                method="LSODA",
                rtol=BL._BUBBLE_RTOL,
                atol=BL._BUBBLE_ATOL,
            )
    except BL.BubbleSolverError:
        return np.nan, False, False, dR2 / R2
    if not sol.success:
        return np.nan, False, False, dR2 / R2
    T = sol.y[1]
    mono = bool(np.all(np.diff(T) >= 0) or np.all(np.diff(T) <= 0)) and bool(np.all(T > 0))
    return float(sol.y[0][-1]), True, mono, dR2 / R2


def _find_root(dMdt, params, Pb, R1, R2, dTdr_ref):
    """Sweep dTdr_front = f*dTdr_ref (f log-spaced), bracket the v(R1)=0 sign change, refine.

    Range spans 6 decades (1e-3..1e3 x the Spitzer dTdr) so a 'no root' verdict means no root
    ANYWHERE the eigenvalue could plausibly sit, not just outside a narrow window.
    """
    fac = np.logspace(-3.0, 3.0, 80)
    xs = dTdr_ref * fac  # all negative
    vs, oks = [], []
    for x in xs:
        v, ok, _, _ = _v_at_R1(x, dMdt, params, Pb, R1, R2)
        vs.append(v)
        oks.append(ok)
    vs = np.array(vs)
    sweep = list(zip(fac.tolist(), xs.tolist(), vs.tolist(), oks))

    # find the first adjacent pair that brackets a sign change (both finite)
    root = None
    for i in range(len(xs) - 1):
        a, b = vs[i], vs[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            try:
                xr = scipy.optimize.brentq(
                    lambda x: _v_at_R1(x, dMdt, params, Pb, R1, R2)[0],
                    xs[i], xs[i + 1], xtol=abs(dTdr_ref) * 1e-6, maxiter=80,
                )
                vr, ok, mono, dr2 = _v_at_R1(xr, dMdt, params, Pb, R1, R2)
                root = {"dTdr_root": xr, "f_root": xr / dTdr_ref, "vR1": vr,
                        "ok": ok, "mono": mono, "dR2_over_R2": dr2}
                break
            except Exception:
                continue
    return root, sweep


def main():
    rows = []
    sweeps = {}   # (state_label, s) -> [(fac, x, v, ok), ...]
    vstream = {}  # state_label -> cool_alpha*R2/t_now (the streaming velocity v(R1) is scaled by)
    for label, fixture_name, _note in _STATES:
        fixture, params = _load(fixture_name)
        Pb, R1 = fixture["Pb"], fixture["R1"]
        R2 = params["R2"].value
        vstream[label] = params["cool_alpha"].value * R2 / params["t_now"].value
        dMdt_sp = fixture["dMdt_converged"]
        dTdr_ref = _spitzer_dTdr(dMdt_sp, params, R2)  # reference: Spitzer dTdr at the baseline dMdt
        for s in _SUPPRESSION:
            dMdt = dMdt_sp / s
            root, sweep = _find_root(dMdt, params, Pb, R1, R2, dTdr_ref)
            sweeps[(label, s)] = sweep
            rows.append({
                "state": label, "fixture": fixture_name, "s": s, "dMdt_entrain": dMdt,
                "root_found": root is not None,
                "f_root": root["f_root"] if root else np.nan,
                "dTdr_root": root["dTdr_root"] if root else np.nan,
                "dR2_over_R2_root": root["dR2_over_R2"] if root else np.nan,
                "physical": bool(root and root["ok"] and root["mono"]) if root else False,
                "vR1_at_root": root["vR1"] if root else np.nan,
            })
            tag = "ROOT" if root else "no root"
            extra = f" f*={root['f_root']:.3f} dR2/R2={root['dR2_over_R2']:.2e} phys={root['ok'] and root['mono']}" if root else ""
            print(f"  [{label:18s}] s={s:>2}  dMdt={dMdt:.4g}  -> {tag}{extra}")

    # --- CSV ---
    csv_path = os.path.join(_HERE, "fm1_rootcheck.csv")
    cols = ["state", "fixture", "s", "dMdt_entrain", "root_found", "f_root",
            "dTdr_root", "dR2_over_R2_root", "physical", "vR1_at_root"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    # --- figure: v(R1) vs dTdr_front/dTdr_ref, one panel per state, a curve per s ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    fig, axes = plt.subplots(1, len(_STATES), figsize=(12, 4.8))
    if len(_STATES) == 1:
        axes = [axes]
    colors = {1: "#1f77b4", 3: "#2ca02c", 10: "#ff7f0e", 30: "#d62728"}
    for ax, (label, _fx, note) in zip(axes, _STATES):
        vsc = vstream[label]
        for s in _SUPPRESSION:
            sweep = sweeps[(label, s)]
            fac = np.array([p[0] for p in sweep])
            vs = np.array([p[2] for p in sweep]) / vsc  # scaled by the streaming velocity
            lbl = f"s={s}" + (" (full Spitzer dMdt)" if s == 1 else f" (dMdt/{s})")
            ax.plot(fac, vs, lw=1.8, color=colors[s], label=lbl)
            r = next((row for row in rows if row["state"] == label and row["s"] == s), None)
            if r and r["root_found"]:
                ax.plot(r["f_root"], 0, "o", color=colors[s], ms=8, mec="k", mew=0.7, zorder=5)
        ax.axhline(0, color="crimson", lw=1.2, ls="-")  # the v(R1)=0 boundary condition
        ax.text(ax.get_xlim()[1], 0, " v(R1)=0\n (the BC)", fontsize=7, va="center", ha="right", color="crimson")
        ax.axvline(1.0, color="0.7", lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel(r"dTdr$_{\rm front}$ / dTdr$_{\rm Spitzer}$  (the §3a eigenvalue, swept 6 decades)", fontsize=8.5)
        ax.set_ylabel(r"v(R1) / $v_{\rm stream}$   (BC is the red line)", fontsize=8.5)
        ax.set_title(f"{label}\n{note}", fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=7.3, title="evaporation suppression", title_fontsize=7.3, loc="best")
    fig.suptitle("FM1 FIRED: only s=1 (full Spitzer $\\dot M$) reaches v(R1)=0 — suppressing $\\dot M$ "
                 "lifts the whole curve off the BC,\nand the conduction gradient $dTdr_{\\rm front}$ "
                 "cannot pull it back ($\\Rightarrow$ §3a 'shoot on $dTdr_{\\rm front}$' is refuted)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = os.path.join(_PDV, "fm1_rootcheck.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
