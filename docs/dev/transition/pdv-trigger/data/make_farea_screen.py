#!/usr/bin/env python3
"""f_area Phase A0 — offline frozen-state screen: is f_κ = f_A = f the area knob? (NO production edit.)

THE CHARTER (`docs/dev/transition/kappa-3way/F_AREA_PLAN.md` §5, Phase A0). f_area is not a new
knob: it is the two shipped knobs driven at EQUAL doses, which the plan's §2.2 derives as the exact
1-D representation of multiplying interface area by f — the layer's T-profile is invariant while
every interface flux (conduction, radiation, evaporation) scales by exactly f. This screen runs the
REAL production solve `get_bubbleproperties_pure` at the two committed captured states with the real
gated params (no monkeypatch, no `trinity/` edit) over f ∈ {1,2,4,8,16} × {κ-only, fA-only,
combined}, and scores the five checks A0.1–A0.5 that the plan PRE-REGISTERED before any of this ran.
GA0 (plan §7) gates the 514-arm bench8 submission: A0.1/A0.2 fail ⇒ STOP.

⚠️ SCOPE — FALSIFIER AND SIGN-CHECK ONLY (`kappa-3way/FINDINGS.md` §12a). No number here is a
full-run calibration. The P1 precedent is exactly this mistake: frozen-state exponents q ≈ 0.55–0.70
did not survive contact with full runs (measured q ≈ 0.27–0.32, `FINDINGS.md` §3). The ONE
principled exception is the §2.2 layer-invariance identity, which is *itself* a per-call statement,
so a captured state tests it in its own regime. The captured states are EARLY snapshots
(θ_snapshot ≈ 0.009 stiff / 0.001 mild), so the θ column is a snapshot, never a blowout θ.

WHAT IS MEASURED per (state, mode, f) — one full solve each, `bubble_dMdt` reset to nan so every
call re-seeds cleanly from Weaver Eq 33 (`_get_init_dMdt`, itself ∝ f_κ^{2/7}):
  * `bubble_dMdt` and its ratio to f=1        — A0.2, and the Ṁ sign that PA3 predicts for full runs
  * `bubble_LTotal` + the L1/L2/L3 split      — A0.3 against the zeroth-order 1 + s·(f−1)
  * max|ΔT/T| vs the f=1 profile of the same state — A0.1, the invariance identity
  * `r2_prime` (Weaver Eq 44 anchor)          — A0.5
  * solver health + the Eq-33 seed vs the converged root — A0.4, and the §9 warm-start hazard
    (the seed scales f^{2/7} while a layer-invariant root scales ≈ f, so the seed must undershoot;
    the size of that undershoot is the evidence gating the ONE candidate `trinity/` edit)

T-PROFILE DEVIATION — the coordinate is stated because the check does not fix one. Primary
(`maxdT_r`) compares T at the same ABSOLUTE radius over the two profiles' overlap, sampled on the
baseline's own 60k grid — no invented grid, no extrapolation. Secondary (`maxdT_z`) repeats it in
depth-from-the-front z = r2_prime − r, which is the coordinate §2.2's identity is literally written
in. Both are reported; A0.1 is scored on the primary and the secondary is the robustness check.

REPRODUCE (from repo root, ~1 min, in-container, no HPC):
    python docs/dev/transition/pdv-trigger/data/make_farea_screen.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/farea_screen.csv   (table=CALL rows + table=CHECK scorecard)
    docs/dev/transition/pdv-trigger/farea_screen.png
"""

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

import trinity.bubble_structure.bubble_luminosity as BL

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the pdv-trigger dir
from _stamp import stamp  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

# Reuse the captured-state loader from the FM1 harness (same dir) — no duplication.
_spec = importlib.util.spec_from_file_location("_fm1", os.path.join(_HERE, "make_fm1_rootcheck.py"))
_fm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm1)

_DOSES = [1, 2, 4, 8, 16]                      # plan §5 Phase A0 grid
_MODES = ["kappa", "fA", "combined"]           # f_κ only / f_A only / the f_area construction
_A0_3_TOL = 0.30                               # A0.3 bar: within +-30% of 1 + s*(f-1)
_A0_4_BAR = 8                                  # A0.4 bar: combined healthy to f >= 8
_A0_4_STOP = 4                                 # A0.4 severe: ceiling < 4 => STOP


def _knobs(mode, f):
    """(f_kappa, f_A) for a mode at dose f. 'combined' is the ONE shared constant."""
    return {"kappa": (f, 1.0), "fA": (1.0, f), "combined": (f, f)}[mode]


def _max_rel_dev(x_base, T_base, x, T):
    """max|ΔT/T| between a run and its f=1 baseline, on the baseline's own points.

    Both profiles are sampled on the baseline abscissa restricted to the two runs' overlap, so
    nothing is extrapolated and no grid is invented. x may be radius (primary) or depth-from-front
    (secondary); both are passed ascending.

    np.unique: the production 60k grid can carry a duplicated radius (the mild state has one, at
    the T = 10^5.5 CIE switch, where T jumps 0.12% across the tie). np.interp resolves a tied
    abscissa to one side arbitrarily, which put a spurious 1.2e-3 floor on every deviation --
    caught by the f1_identity check below, which requires a profile to deviate from ITSELF by
    exactly 0. Keeping the first occurrence removes the tie without touching the physics.
    """
    x_base, ib = np.unique(x_base, return_index=True)
    x, i = np.unique(x, return_index=True)
    T_base, T = T_base[ib], T[i]
    lo, hi = max(x_base[0], x[0]), min(x_base[-1], x[-1])
    m = (x_base >= lo) & (x_base <= hi)
    if not np.any(m) or not np.all(np.isfinite(T)):
        return float("nan")
    return float(np.max(np.abs(np.interp(x_base[m], x, T) - T_base[m]) / T_base[m]))


def _run(params, fk, fa):
    """Full production solve with cooling_boost_kappa=fk and cooling_boost_fA=fa.

    Sets `.value` directly, exactly as make_fkappa_leverage.py does for f_κ: the five runtime read
    sites (F_AREA_PLAN §4) all read `.value`, so this is production behaviour. The load-time
    cross-knob warning (`registry.py::_validate_cooling_boost_fA`) is a read_param-time provenance
    line for bench8 and is not exercised here.
    """
    params["cooling_boost_kappa"].value = float(fk)
    params["cooling_boost_fA"].value = float(fa)
    params["bubble_dMdt"].value = float("nan")  # clean fsolve from the Eq-33 seed each time
    try:
        bp = BL.get_bubbleproperties_pure(params)
    except Exception as e:  # a failed solve is data (A0.4), not a crash
        print(f"      (f_κ={fk} f_A={fa}: solve failed: {type(e).__name__})")
        return None
    T = np.asarray(bp.bubble_T_arr, dtype=float)
    r = np.asarray(bp.bubble_r_arr, dtype=float)
    mono = bool(np.all(np.diff(T) >= 0) or np.all(np.diff(T) <= 0)) and bool(np.all(T > 0))
    order = np.argsort(r)
    r2_prime = float(np.asarray(
        BL._get_bubble_ODE_initial_conditions(bp.bubble_dMdt, params, bp.Pb, bp.R1)[0]).item())
    return {
        "dMdt": float(bp.bubble_dMdt),
        "dMdt_seed": float(BL._get_init_dMdt(params, bp.Pb)),
        "residual_abs": abs(float(BL._get_velocity_residuals(bp.bubble_dMdt, params, bp.Pb, bp.R1))),
        "LTotal": float(bp.bubble_LTotal),
        "L1": float(bp.bubble_L1Bubble),
        "L2": float(bp.bubble_L2Conduction),
        "L3": float(bp.bubble_L3Intermediate),
        "r2_prime": r2_prime,
        # dR2 by subtraction: at the stiff state's dR2/R2 ~ 1e-10 that keeps ~6 significant
        # digits, which is ample for A0.5's ranking (and for the ratio column).
        "dR2": float(params["R2"].value) - r2_prime,
        "npts": int(T.size),
        "healthy": bool(np.isfinite(bp.bubble_dMdt) and bp.bubble_dMdt > 0
                        and np.isfinite(bp.bubble_LTotal) and bp.bubble_LTotal > 0 and mono),
        "_r": r[order], "_T": T[order],
    }


def _verdict(ok):
    return "PASS" if ok else "FAIL"


def main():
    rows, checks = [], []
    res = {}  # (state, mode, f) -> measurement dict
    base = {}  # state -> f=1 measurement (identical in all three modes; asserted below)

    for label, fixture_name, _note in _fm1._STATES:
        _fixture, params = _fm1._load(fixture_name)
        assert "cooling_boost_kappa" in params and "cooling_boost_fA" in params, \
            f"{fixture_name}: captured state lacks a boost knob — the screen would silently no-op"
        Lmech = params["Lmech_total"].value
        R2 = params["R2"].value
        print(f"[{label}] R2={R2:.6g} pc  Lmech={Lmech:.4g}")
        for mode in _MODES:
            for f in _DOSES:
                fk, fa = _knobs(mode, f)
                m = _run(params, fk, fa)
                res[(label, mode, f)] = m
                if f == 1 and mode == _MODES[0]:
                    assert m is not None, f"{label}: the f=1 baseline solve failed — screen is void"
                    base[label] = m
                if m is None:
                    rows.append({"table": "CALL", "state": label, "mode": mode, "f": f,
                                 "f_kappa": fk, "f_A": fa, "healthy": False})
                    continue
                b = base[label]
                zb, z = b["r2_prime"] - b["_r"][::-1], m["r2_prime"] - m["_r"][::-1]
                rows.append({
                    "table": "CALL", "state": label, "mode": mode, "f": f, "f_kappa": fk, "f_A": fa,
                    "dMdt": m["dMdt"], "dMdt_ratio": m["dMdt"] / b["dMdt"],
                    "dMdt_seed": m["dMdt_seed"], "dMdt_over_seed": m["dMdt"] / m["dMdt_seed"],
                    "residual_abs": m["residual_abs"],
                    "LTotal": m["LTotal"], "LTotal_ratio": m["LTotal"] / b["LTotal"],
                    "L1": m["L1"], "L2": m["L2"], "L3": m["L3"],
                    "iface_share": (m["L2"] + m["L3"]) / m["LTotal"],
                    "theta_snapshot": m["LTotal"] / Lmech,
                    "r2_prime": m["r2_prime"], "dR2": m["dR2"],
                    "dr2_prime_rel": abs(m["r2_prime"] - b["r2_prime"]) / b["r2_prime"],
                    "dR2_ratio": m["dR2"] / b["dR2"],
                    "maxdT_r": _max_rel_dev(b["_r"], b["_T"], m["_r"], m["_T"]),
                    "maxdT_z": _max_rel_dev(zb, b["_T"][::-1], z, m["_T"][::-1]),
                    "npts": m["npts"], "healthy": m["healthy"],
                })
                print(f"      {mode:>8} f={f:>3}  Ṁ×{m['dMdt'] / b['dMdt']:.4f}  "
                      f"L×{m['LTotal'] / b['LTotal']:.4f}  max|ΔT/T|={rows[-1]['maxdT_r']:.3e}  "
                      f"healthy={m['healthy']}")

    # ---- the f=1 identity (the A0-local analogue of GA6, and the harness's self-check) ----
    # All three modes at f=1 are the same call (both knobs gated), and a profile must deviate from
    # ITSELF by exactly 0 -- that second half is what caught the duplicated-abscissa artifact.
    for label, _fx, _n in _fm1._STATES:
        b = base[label]
        v = {mode: res[(label, mode, 1)]["dMdt"] for mode in _MODES}
        self_dev = max(_max_rel_dev(b["_r"], b["_T"], res[(label, mode, 1)]["_r"],
                                    res[(label, mode, 1)]["_T"]) for mode in _MODES)
        checks.append({"table": "CHECK", "check": "f1_identity", "state": label, "f": 1,
                       "value": max(v.values()) - min(v.values()), "reference": 0.0,
                       "verdict": _verdict(len(set(v.values())) == 1 and self_dev == 0.0),
                       "note": f"same dMdt in all 3 modes and zero self-deviation (max {self_dev:g})"})

    dosed = [f for f in _DOSES if f != 1]

    def _pair(label, f, key):
        c, k = res[(label, "combined", f)], res[(label, "kappa", f)]
        if c is None or k is None or not (c["healthy"] and k["healthy"]):
            return None
        return c[key], k[key]

    # ---- A0.1 T-profile invariance ranking: combined deviates LESS than kappa-only ----
    for label, _fx, _n in _fm1._STATES:
        b = base[label]
        for f in dosed:
            c, k = res[(label, "combined", f)], res[(label, "kappa", f)]
            if c is None or k is None or not (c["healthy"] and k["healthy"]):
                checks.append({"table": "CHECK", "check": "A0.1", "state": label, "f": f,
                               "verdict": "SKIP", "note": "a mode did not solve healthily"})
                continue
            dc = _max_rel_dev(b["_r"], b["_T"], c["_r"], c["_T"])
            dk = _max_rel_dev(b["_r"], b["_T"], k["_r"], k["_T"])
            checks.append({"table": "CHECK", "check": "A0.1", "state": label, "f": f,
                           "value": dc, "reference": dk, "verdict": _verdict(dc < dk),
                           "note": "max|dT/T| combined vs kappa-only (absolute-r overlap)"})

    # ---- A0.2 per-call Ṁ superadditivity: combined ratio > kappa-only ratio ----
    for label, _fx, _n in _fm1._STATES:
        b = base[label]
        for f in dosed:
            p = _pair(label, f, "dMdt")
            if p is None:
                checks.append({"table": "CHECK", "check": "A0.2", "state": label, "f": f,
                               "verdict": "SKIP", "note": "a mode did not solve healthily"})
                continue
            rc, rk = p[0] / b["dMdt"], p[1] / b["dMdt"]
            checks.append({"table": "CHECK", "check": "A0.2", "state": label, "f": f,
                           "value": rc, "reference": rk, "verdict": _verdict(rc > rk),
                           "note": f"Mdot ratio combined vs kappa-only; layer-invariance ref f={f}"})

    # ---- A0.3 L_total linearity vs 1 + s*(f-1), s from the same state's f=1 call ----
    for label, _fx, _n in _fm1._STATES:
        b = base[label]
        s = (b["L2"] + b["L3"]) / b["LTotal"]
        for f in dosed:
            c = res[(label, "combined", f)]
            if c is None or not c["healthy"]:
                checks.append({"table": "CHECK", "check": "A0.3", "state": label, "f": f,
                               "verdict": "SKIP", "note": "combined did not solve healthily"})
                continue
            model = 1 + s * (f - 1)
            meas = c["LTotal"] / b["LTotal"]
            checks.append({"table": "CHECK", "check": "A0.3", "state": label, "f": f,
                           "value": meas, "reference": model,
                           "verdict": _verdict(abs(meas / model - 1) <= _A0_3_TOL),
                           "note": f"s={s:.4f}; bar |meas/model-1| <= {_A0_3_TOL}"})

    # ---- A0.4 viability ceiling of the combined solve ----
    for label, _fx, _n in _fm1._STATES:
        ok = [f for f in _DOSES if res[(label, "combined", f)] is not None
              and res[(label, "combined", f)]["healthy"]]
        ceil = max(ok) if ok else 0
        checks.append({"table": "CHECK", "check": "A0.4", "state": label, "f": ceil,
                       "value": ceil, "reference": _A0_4_BAR,
                       "verdict": _verdict(ceil >= _A0_4_BAR),
                       "note": f"combined healthy up to f={ceil}; STOP bar is ceiling < {_A0_4_STOP}"})

    # ---- A0.5 anchor invariance: |Δr2_prime|(combined) < |Δr2_prime|(kappa-only) ----
    for label, _fx, _n in _fm1._STATES:
        b = base[label]
        for f in dosed:
            p = _pair(label, f, "r2_prime")
            if p is None:
                checks.append({"table": "CHECK", "check": "A0.5", "state": label, "f": f,
                               "verdict": "SKIP", "note": "a mode did not solve healthily"})
                continue
            dc, dk = abs(p[0] - b["r2_prime"]), abs(p[1] - b["r2_prime"])
            checks.append({"table": "CHECK", "check": "A0.5", "state": label, "f": f,
                           "value": dc, "reference": dk, "verdict": _verdict(dc < dk),
                           "note": "|delta r2_prime| combined vs kappa-only (Weaver Eq 44 anchor)"})

    # ---- CSV ----
    cols = ["table", "check", "state", "mode", "f", "f_kappa", "f_A", "dMdt", "dMdt_ratio",
            "dMdt_seed", "dMdt_over_seed", "residual_abs", "LTotal", "LTotal_ratio", "L1", "L2",
            "L3", "iface_share", "theta_snapshot", "r2_prime", "dR2", "dr2_prime_rel", "dR2_ratio",
            "maxdT_r", "maxdT_z", "npts", "healthy", "value", "reference", "verdict", "note"]
    csv_path = os.path.join(_HERE, "farea_screen.csv")
    with open(csv_path, "w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# f_area Phase A0 offline screen (kappa-3way/F_AREA_PLAN.md §5): the production solve at"
                 " two committed captured states, f in {1,2,4,8,16} x {kappa-only, fA-only, combined}.\n")
        fh.write("# table=CALL: one row per solve. table=CHECK: the PRE-REGISTERED A0.1-A0.5 scorecard"
                 " (frozen in the plan before this ran; scored, never retuned).\n")
        fh.write("# SCOPE: falsifier and sign-check ONLY (FINDINGS.md §12a) -- no number here is a"
                 " full-run calibration; the §2.2 invariance identity is the one per-call exception.\n")
        fh.write("# Regenerate: python docs/dev/transition/pdv-trigger/data/make_farea_screen.py\n")
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        w.writerows(checks)
    print(f"wrote {csv_path}")

    # ---- scorecard + GA0 ----
    print("\n=== A0 SCORECARD (pre-registered, F_AREA_PLAN §5) ===")
    tally = {}
    for c in checks:
        if c["check"] == "f1_identity":
            continue
        p, f_, s_ = tally.setdefault(c["check"], [0, 0, 0])
        tally[c["check"]] = [p + (c["verdict"] == "PASS"), f_ + (c["verdict"] == "FAIL"),
                             s_ + (c["verdict"] == "SKIP")]
    for name in ["A0.1", "A0.2", "A0.3", "A0.4", "A0.5"]:
        p, f_, s_ = tally.get(name, [0, 0, 0])
        print(f"  {name}: {p} PASS / {f_} FAIL / {s_} SKIP  -> {'PASS' if f_ == 0 else 'FAIL'}")
    failed = {n for n in tally if tally[n][1] > 0}
    ceilings = [c["value"] for c in checks if c["check"] == "A0.4"]
    severe = ({"A0.1", "A0.2"} & failed) or any(v < _A0_4_STOP for v in ceilings)
    print("GA0: " + ("STOP — do not submit bench8" if severe else
                     ("PASS" if not failed else f"PASS with flags {sorted(failed)}")))

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return
    plt.rcParams["text.usetex"] = False
    colr = {"kappa": "#d62728", "fA": "#1f77b4", "combined": "#2ca02c"}
    mark = {"kappa": "o", "fA": "s", "combined": "D"}
    name = {"kappa": r"$f_\kappa$ only", "fA": r"$f_A$ only", "combined": r"combined ($f_\kappa=f_A=f$)"}
    fd = np.array(dosed, float)
    fig, axes = plt.subplots(len(_fm1._STATES), 4, figsize=(17.5, 8.4))

    def _series(label, mode, key):
        out = []
        for f in dosed:
            m = res[(label, mode, f)]
            out.append(m[key] if (m is not None and m["healthy"]) else np.nan)
        return np.array(out, float)

    def _draw(ax, mode, y):
        """kappa-only goes down thick and semi-transparent; combined rides on top with markers.

        The two are within ~1% on three of the four panels -- the result -- so a plain overplot
        would simply hide the red curve under the green one.
        """
        wide = mode == "kappa"
        ax.plot(fd, y, mark[mode] + ("--" if mode != "combined" else "-"), color=colr[mode],
                lw=4.0 if wide else 1.8, ms=9 if wide else 6, alpha=0.45 if wide else 1.0,
                zorder=2 if wide else 3, label=name[mode])

    def _title(ax, text, check, label):
        v = [c["verdict"] for c in checks if c["check"] == check and c["state"] == label]
        tag = f"{v.count('PASS')} PASS / {v.count('FAIL')} FAIL"
        ax.set_title(f"{text}\n→ {tag}", fontsize=9.5, fontweight="bold",
                     color="#8b0000" if v.count("FAIL") else "#1c6b1c")

    for row, (label, _fx, note) in enumerate(_fm1._STATES):
        b = base[label]
        s = (b["L2"] + b["L3"]) / b["LTotal"]
        ax0, ax1, ax2, ax3 = axes[row]

        for mode in _MODES:
            _draw(ax0, mode, [_max_rel_dev(b["_r"], b["_T"], res[(label, mode, f)]["_r"],
                                           res[(label, mode, f)]["_T"])
                              if (res[(label, mode, f)] is not None
                                  and res[(label, mode, f)]["healthy"]) else np.nan
                              for f in dosed])
        ax0.set_yscale("log")
        ax0.set_ylabel(r"max$|\Delta T/T|$ vs $f=1$", fontsize=9)
        _title(ax0, "A0.1 — layer invariance\n(bar: combined BELOW $f_\\kappa$-only)", "A0.1", label)

        ax1.plot(fd, fd, ":", color="0.25", lw=1.6, zorder=1,
                 label=r"$f^{1}$ — what area multiplication predicts")
        ax1.plot(fd, fd ** (2 / 7), "-", color="0.55", lw=3.5, zorder=1,
                 label=r"$f^{2/7}$ — the Weaver/Eq-47 conduction channel")
        ax1.axhline(1.0, color="0.4", lw=1.0, ls=":", zorder=1)
        for mode in _MODES:
            _draw(ax1, mode, _series(label, mode, "dMdt") / b["dMdt"])
        ax1.set_yscale("log")
        ax1.set_ylabel(r"$\dot M(f)/\dot M(1)$", fontsize=9)
        _title(ax1, "A0.2 — mass-loading superadditivity\n(bar: combined ABOVE $f_\\kappa$-only)",
               "A0.2", label)

        model = 1 + s * (fd - 1)
        ax2.fill_between(fd, model * (1 - _A0_3_TOL), model * (1 + _A0_3_TOL),
                         color="#2ca02c", alpha=0.14, zorder=1)
        ax2.plot(fd, model, ":", color="#1c6b1c", lw=1.6, zorder=1,
                 label=rf"$1+s(f-1)$, $s={s:.3f}$ (±{int(_A0_3_TOL * 100)}%)")
        for mode in _MODES:
            _draw(ax2, mode, _series(label, mode, "LTotal") / b["LTotal"])
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$L_{\rm cool}(f)/L_{\rm cool}(1)$", fontsize=9)
        _title(ax2, "A0.3 — $L$ bookkeeping\n(bar: combined inside the band)", "A0.3", label)

        for mode in _MODES:
            _draw(ax3, mode, np.abs(_series(label, mode, "r2_prime") - b["r2_prime"]) / b["r2_prime"])
        ax3.set_yscale("log")
        ax3.set_ylabel(r"$|\Delta r_2'|/r_2'$", fontsize=9)
        _title(ax3, "A0.5 — Eq-44 anchor invariance\n(bar: combined BELOW $f_\\kappa$-only)",
               "A0.5", label)

        for ax in (ax0, ax1, ax2, ax3):
            ax.set_xscale("log", base=2)
            ax.set_xticks(dosed)
            ax.set_xticklabels([str(f) for f in dosed])
            ax.set_xlabel("dose $f$", fontsize=9)
            ax.legend(fontsize=7.0, loc="best")
            ax.tick_params(labelsize=8)
        ax0.text(-0.40, 0.5, f"{label}\n{note}", transform=ax0.transAxes, rotation=90,
                 va="center", ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("f_area Phase A0 — offline frozen-state screen of the combined knob "
                 "($f_\\kappa=f_A=f$) at two captured states.  "
                 "FALSIFIER / SIGN-CHECK ONLY — no number here is a full-run calibration.",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.945, "The combined knob tracks $f_\\kappa$ alone: $\\dot M\\propto f^{2/7}$, "
             "not the $f^{1}$ that area multiplication requires — so the Eq-44 front anchor "
             "$dR_2\\propto f_\\kappa/\\dot M$ thickens as $f^{5/7}$ and the layer cannot stay invariant.",
             ha="center", fontsize=9.2, style="italic", color="#8b0000")
    fig.tight_layout(rect=(0.02, 0, 1, 0.925))
    png = os.path.join(_PDV, "farea_screen.png")
    fig.savefig(png, dpi=135)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
