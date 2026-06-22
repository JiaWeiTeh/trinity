#!/usr/bin/env python3
"""C0 substrate certification + physical-anchor harness (clean-room; see PLAN.md).

Two things, both computed per-snapshot so we can inspect them BY PHASE and BY TIME
(never trust a single summary number -- behaviour differs across config / phase /
feedback surge):

  C0.2  beta/delta <-> trajectory consistency (INTERNAL, code-derived).
        The energy ODE is the Rahner A12 form (get_betadelta.py:182): cooling
        enters via beta, not an explicit Lloss term. So we certify the DEFINITIONS
        the code enforces:
            beta  = -(t/Pb)(dPb/dt)  => predicted dPb/dt = -beta*Pb/t   (:248)
            delta =  (t/T )(dT /dt)  => predicted dT0/dt =  delta*T0/t   (:294)
        finite-differencing stored Pb(t), T0(t) over implicit-phase snapshots.
        NOTE (open item, PLAN.md S2): the delta<->T0 check may be TAUTOLOGICAL if
        T0 is advanced by that same ODE -- reported but flagged, not yet trusted.

  f_ret PHYSICAL ANCHOR (EXTERNAL, PLAN.md S0.1).  Retained hot-bubble energy
        fraction f_ret(t) = Eb / integral(Lmech_total dt)  (left-rectangle).
        Compare to the 3D-sim/observation band ~0.01-0.1, DECREASING with time
        (Lancaster+2021; Geen+2021 ~1%). This is the test of whether the stall is
        a trigger problem (f_ret reaches the band) or an under-cooling physics gap
        (f_ret stays far above it). Independent of TRINITY/Weaver being correct.

Pure diagnostic: nothing in trinity/ is modified; production is untouched.

Usage:
    python c0_consistency.py <dictionary.jsonl>            # analyze an existing run
    python c0_consistency.py <config.param> --stop-t 5 --out data/foo.csv  # run+analyze
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

PHASES = ("energy", "implicit", "transition", "momentum")
LIT_BAND = (0.01, 0.10)  # Lancaster+2021 / Geen+2021 retained-energy band


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:
        return "unknown"


def _finite(*vals) -> bool:
    return all(v is not None and v == v and abs(v) != float("inf") for v in vals)


def load_rows(jsonl_path: str) -> list[dict]:
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda d: d.get("t_now", 0.0))
    return rows


def annotate(rows: list[dict]) -> list[dict]:
    """Add cumulative injected energy E_inj and retained fraction f_ret to each row,
    plus the per-pair consistency residuals res_beta / res_delta (implicit only)."""
    E_inj = 0.0
    out = []
    for i, r in enumerate(rows):
        rec = {
            "t_now": r.get("t_now"),
            "phase": r.get("current_phase"),
            "Eb": r.get("Eb"),
            "Lmech_total": r.get("Lmech_total"),
            "Pb": r.get("Pb"),
            "T0": r.get("T0"),
            "cool_beta": r.get("cool_beta"),
            "cool_delta": r.get("cool_delta"),
            "R2": r.get("R2"),
            "v2": r.get("v2"),
            "bubble_Lgain": r.get("bubble_Lgain"),
            "bubble_Lloss": r.get("bubble_Lloss"),
            "bubble_T_r_Tb": r.get("bubble_T_r_Tb"),
            "betadelta_converged": r.get("betadelta_converged"),
            # H0 trigger-harvest columns (candidate families F0-F4 + the Eb-peak oracle)
            "Lmech_W": r.get("Lmech_W"),
            "Lmech_SN": r.get("Lmech_SN"),
            "R1": r.get("R1"),
            "rCloud": r.get("rCloud"),
            "F_ram": r.get("F_ram"),
            "F_rad": r.get("F_rad"),
            "F_grav": r.get("F_grav"),
            "F_ISM": r.get("F_ISM"),
            "P_HII": r.get("P_HII"),
            "P_ram": r.get("P_ram"),
            "P_drive": r.get("P_drive"),
        }
        # GENUINE delta-side check: ODE-integrated T0 vs structure-solved T(xi_Tb).
        # (delta<->dT0/dt is tautological because T0 is advanced by that ODE; but the
        #  bubble structure solves T independently, so T0 must match bubble_T_r_Tb.)
        _T0, _Ts = r.get("T0"), r.get("bubble_T_r_Tb")
        rec["res_T0_struct"] = (abs(_T0 - _Ts) / abs(_Ts)
                                if _finite(_T0, _Ts) and abs(_Ts) > 0 else None)
        # f_ret = Eb / E_inj, E_inj cumulative left-rectangle of Lmech_total
        rec["E_inj"] = E_inj if E_inj > 0 else None
        Eb = r.get("Eb")
        rec["f_ret"] = (Eb / E_inj) if (E_inj > 0 and _finite(Eb)) else None
        # advance E_inj for the NEXT row (left-rectangle: state before segment)
        if i + 1 < len(rows):
            dt = rows[i + 1].get("t_now", 0.0) - r.get("t_now", 0.0)
            Lm = r.get("Lmech_total")
            if _finite(Lm, dt) and dt > 0:
                E_inj += Lm * dt
        out.append(rec)

    # consistency residuals on adjacent IMPLICIT-phase rows
    for a, b in zip(out, out[1:]):
        if a["phase"] != "implicit":
            continue
        t0, t1 = a["t_now"], b["t_now"]
        if not _finite(t0, t1) or t1 <= t0 or t0 <= 0:
            continue
        dt = t1 - t0
        Pb0, Pb1, beta0 = a["Pb"], b["Pb"], a["cool_beta"]
        if _finite(Pb0, Pb1, beta0) and abs(Pb0) > 0:
            meas, pred = (Pb1 - Pb0) / dt, -beta0 * Pb0 / t0
            a["res_beta"] = abs(meas - pred) / (abs(pred) + 1e-300)
        T0_, T1_, delta0 = a["T0"], b["T0"], a["cool_delta"]
        if _finite(T0_, T1_, delta0) and abs(T0_) > 0:
            meas, pred = (T1_ - T0_) / dt, delta0 * T0_ / t0
            a["res_delta"] = abs(meas - pred) / (abs(pred) + 1e-300)
    return out


def _stats(xs: list[float]) -> dict | None:
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(s), "median": statistics.median(s),
            "p90": s[min(len(s) - 1, int(0.9 * len(s)))], "max": max(s), "min": min(s)}


def summarize(rows: list[dict], provenance: str) -> None:
    print(f"# C0 certify + f_ret anchor")
    print(f"# {provenance}")
    from collections import Counter
    print(f"# phase rows: {dict(Counter(r['phase'] for r in rows))}")

    # --- C0.2 consistency, overall AND time-resolved (early/mid/late implicit) ---
    impl = [r for r in rows if r["phase"] == "implicit"]
    conv = [r for r in impl if r.get("betadelta_converged")]
    print("## C0.2 substrate consistency (implicit phase)")
    print(f"  betadelta_converged: {len(conv)}/{len(impl)} implicit segments")
    # GENUINE trajectory gate: beta the solver returned vs how Pb ACTUALLY evolved.
    s = _stats([r.get("res_beta") for r in impl])
    if s:
        bar = "PASS" if s["median"] <= 0.05 else "FAIL"
        print(f"  beta<->dPb/dt   (GENUINE trajectory) : n={s['n']:4d} median={s['median']:.3%} "
              f"p90={s['p90']:.3%} max={s['max']:.3%} [median<=5%:{bar}]")
    # solver's own T-residual |T0-T_struct|/T_struct, on CONVERGED segments only
    # (T0 and bubble_T_r_Tb are the two sides of the solver's T_residual, :449).
    s = _stats([r.get("res_T0_struct") for r in (conv or impl)])
    if s:
        where = "converged" if conv else "ALL impl (NO conv flag!)"
        print(f"  |T0-Tstruct|/Tstruct (solver T-resid, {where}): n={s['n']:4d} "
              f"median={s['median']:.3%} p90={s['p90']:.3%} max={s['max']:.3%}")
    # tautological delta check (diagnostic only; expect ~0)
    s = _stats([r.get("res_delta") for r in impl])
    if s:
        print(f"  delta<->dT0/dt  (TAUTOLOGICAL, diag)  : median={s['median']:.3%} (expect ~0)")
    # time-resolved: thirds of the implicit phase (catch surge-localized breakdown)
    if len(impl) >= 6:
        third = len(impl) // 3
        for name, seg in [("early", impl[:third]), ("mid", impl[third:2 * third]),
                          ("late", impl[2 * third:])]:
            tlo, thi = seg[0]["t_now"], seg[-1]["t_now"]
            parts = []
            for key in ("res_beta", "res_T0_struct"):
                sb = _stats([r.get(key) for r in seg])
                if sb:
                    parts.append(f"{key}:med={sb['median']:.2%},max={sb['max']:.2%}")
            if parts:
                print(f"    {name:5s} [t={tlo:.3f}-{thi:.3f}]: " + "  ".join(parts))

    # --- f_ret physical anchor, per phase + trend ---
    print(f"## f_ret = Eb/integral(Lmech dt)  vs literature band {LIT_BAND} (decreasing)")
    for ph in PHASES:
        seg = [r for r in rows if r["phase"] == ph and r["f_ret"] is not None]
        s = _stats([r["f_ret"] for r in seg])
        if not s:
            continue
        f_first, f_last = seg[0]["f_ret"], seg[-1]["f_ret"]
        trend = "DOWN" if f_last < f_first else "UP"
        in_band = "IN-BAND" if LIT_BAND[0] <= f_last <= LIT_BAND[1] else (
            "ABOVE(under-cooled?)" if f_last > LIT_BAND[1] else "BELOW")
        print(f"  {ph:11s}: n={s['n']:4d} f_ret {f_first:.3g}->{f_last:.3g} ({trend}) "
              f"min={s['min']:.3g} | end {in_band}")


class _FrozenInterp:
    """Wraps an SPS interpolator so it returns its t_freeze value for ANY input t,
    while preserving .x (update_feedback uses it for the SPS time-range check). Freezing
    at the interpolator SOURCE means EVERY consumer sees the same constant -- both
    get_current_sps_feedback AND the paths that read params['sps_f'] directly
    (get_InitPhaseParam:88, the snapshot logger), which a get_current_sps_feedback
    rebind alone misses."""
    __slots__ = ("_v", "x")

    def __init__(self, f, tf):
        self._v = f(tf)                  # 0-d array at the freeze time (supports [()])
        self.x = getattr(f, "x", None)

    def __call__(self, t):
        return self._v


def _freeze_feedback(t_freeze: float) -> None:
    """Freeze ALL stellar feedback to its t_freeze value for the whole run, by replacing
    the SPS interpolators with constant ones at their source -- read_sps.get_interpolation,
    which main() calls once during setup (main.py:148) BEFORE any phase runs. Every
    SPSFeedback field (Lmech_W/SN/total, pdot_*, v_mech_total, Qi/Li/Ln/Lbol) becomes a
    true constant; derived rates (v_mech = 2L/pdot) stay self-consistent, and the
    finite-difference pdotdot_total -> 0 (correct for constant feedback). Cumulative
    budgets (E_inj, shell mass, momentum) keep integrating the now-constant rates
    downstream. This is the robust freeze: a get_current_sps_feedback monkeypatch only
    covered SOME paths (others read params['sps_f'] directly), so the sim still saw real
    feedback -- verified by the smoke's bubble_Lgain being real, not frozen."""
    import trinity.sps.read_sps as _rs
    _orig = _rs.get_interpolation

    def frozen_get_interpolation(sps_data, _tf=t_freeze, _o=_orig):
        d = _o(sps_data)
        return {k: (_FrozenInterp(f, _tf) if callable(f) and hasattr(f, "x") else f)
                for k, f in d.items()}

    _rs.get_interpolation = frozen_get_interpolation


def run_config(param_path: str, stop_t: float | None, refine: float = 1.0,
               solver: str = "hybr", freeze_feedback_at: float | None = None,
               run_dir: str | None = None) -> str:
    # The harness calls start_expansion() directly (not via run.py), which trips
    # main.py's DEBUG-logging fallback -- per-RHS DEBUG records are a measured
    # hot-path cost over a full run (registry log_level note). Install an INFO
    # handler FIRST so the fallback (`if not root.handlers`) is skipped. INFO is
    # the normal run.py operating point: as fast as WARNING but per-phase insightful.
    import logging
    logging.basicConfig(level=logging.INFO)
    from trinity._input import read_param
    from trinity import main as trinity_main
    # C0.2 bar (ii) refinement check: shrink the adaptive-timestep scales by `refine`
    # so snapshots are denser. If res_beta drops ~proportionally, the residual is
    # finite-difference TRUNCATION (∝ Δt), not a substrate defect. Monkeypatch the
    # module constants only -- nothing in trinity/ is edited on disk.
    if refine and refine != 1.0:
        import trinity.phase1b_energy_implicit.run_energy_implicit_phase as rmod
        # ADAPTIVE_THRESHOLD_DEX is the DOMINANT timestep control in early implicit
        # (the adaptive max_dex controller, not the DT_SEGMENT caps) -- tighten it too
        # or the knob barely bites where res_beta truncation is worst.
        for c in ("DT_SEGMENT_INIT", "DT_SEGMENT_MIN", "DT_SEGMENT_MAX",
                  "ODE_MAX_STEP", "DT_SEGMENT_COLLAPSE", "ADAPTIVE_THRESHOLD_DEX"):
            if hasattr(rmod, c):
                setattr(rmod, c, getattr(rmod, c) / refine)
    params = read_param.read_param(param_path)
    # Persistent run_dir (survives container soft-restart; lets a supervisor read the
    # live dictionary.jsonl mid-run for checkpointing). Default: ephemeral mkdtemp.
    if run_dir is not None:
        out_dir = run_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    else:
        out_dir = tempfile.mkdtemp(prefix="c0_")
    params["path2output"].value = out_dir
    params["betadelta_solver"].value = solver
    ll = params.get("log_level", None)
    if ll is not None and hasattr(ll, "value"):
        ll.value = "INFO"
    if stop_t is not None:
        params["stop_t"].value = stop_t
    # Opt-in: freeze stellar feedback to its t_freeze value for the whole run.
    # NO-OP when freeze_feedback_at is None -- behaviour is byte-identical to before.
    if freeze_feedback_at is not None:
        _freeze_feedback(freeze_feedback_at)
    try:
        trinity_main.start_expansion(params)
    except SystemExit:
        pass
    hits = list(Path(out_dir).rglob("dictionary.jsonl"))
    if not hits:
        sys.exit(f"no dictionary.jsonl produced under {out_dir}")
    # newest, in case a reused run_dir holds dictionaries from prior attempts
    return str(max(hits, key=lambda p: p.stat().st_mtime))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help=".param to run (hybr), or an existing dictionary.jsonl")
    ap.add_argument("--stop-t", type=float, default=None)
    ap.add_argument("--refine", type=float, default=1.0,
                    help="shrink adaptive-timestep scales by this factor (C0.2 refinement check)")
    ap.add_argument("--solver", default="hybr", choices=["hybr", "legacy"],
                    help="betadelta solver (default hybr; 'legacy' for the BEFORE comparison)")
    ap.add_argument("--out", default=None, help="write full per-row CSV here")
    ap.add_argument("--freeze-feedback-at", type=float, default=None, metavar="MYR",
                    help="freeze ALL stellar feedback to its value at this time [Myr], held "
                         "constant for the whole run (no WR/SN surges). Default: off (no-op).")
    ap.add_argument("--run-dir", default=None, metavar="DIR",
                    help="persistent output dir for the run (default: ephemeral mkdtemp). Use a "
                         "stable path so a supervisor can checkpoint the live dictionary.jsonl.")
    args = ap.parse_args()

    if args.target.endswith(".jsonl"):
        jsonl, prov = args.target, f"snapshots {args.target} (provenance not certified)"
    else:
        jsonl = run_config(args.target, args.stop_t, args.refine, args.solver,
                           args.freeze_feedback_at, args.run_dir)
        frz = f", freeze_feedback_at={args.freeze_feedback_at}" if args.freeze_feedback_at is not None else ""
        prov = f"ran {args.target} ({args.solver}, stop_t={args.stop_t}, refine={args.refine}{frz}) @ {_git_sha()}"

    rows = annotate(load_rows(jsonl))
    summarize(rows, prov)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["t_now", "phase", "betadelta_converged", "Eb", "Lmech_total",
                "Lmech_W", "Lmech_SN", "E_inj", "f_ret", "bubble_Lgain", "bubble_Lloss",
                "Pb", "R1", "R2", "v2", "rCloud",
                "F_ram", "F_rad", "F_grav", "F_ISM", "P_HII", "P_ram", "P_drive",
                "cool_beta", "res_beta", "T0", "bubble_T_r_Tb", "res_T0_struct",
                "cool_delta", "res_delta"]
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"# wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
