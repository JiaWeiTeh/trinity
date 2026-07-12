#!/usr/bin/env python3
"""bench5 Phase-5 analysis — fire map + θ(t) tracks + El-Badry cross-check (PROVISIONAL, in-container).

Reads the committed campaign artifacts (runs/data/bench5_summary.csv + runs/data/bench5_traj/*.csv)
and the registered sim-free prediction (data/bench5_elbadry_prediction.csv), and produces the
Phase-5 reads that the COMPLETED arms support. Regenerable — rerun as more arms land.

WHAT THE IN-CONTAINER RUN SUPPORTS (honest scope, SOURCE_TERM_DESIGN §3 Phase 5):
  • FIRE MAP from the PRODUCTION arms: per (bench, f_A) — did cooling_balance fire, at what t, θ_max,
    and fate (shell_collapsed / stopping_time / momentum). This IS a real result.
  • θ(t) up to fire (the implicit-phase trajectory is censored at the transition for a prod arm).
  • El-Badry cross-check: θ_max vs the registered θ_EB(λδv=3, n̄) and the L21b Θ band [0.90, 0.99].
WHAT IT DOES NOT SUPPORT (⇒ HPC-deferred, stated in the output + FINDINGS §15h):
  • Θ_cum over the Lancaster window [t_first, min(3 Myr, …)] and the |Δlog(1−θ)| dex metric — these
    need the DIAGNOSTIC arms (transition_trigger=blowout: uncensored θ(t) through the whole energy
    phase). No diagnostic arm completed in-container (they are the slowest, most restart-vulnerable).
    Θ_cum computed here is over the PRE-FIRE window only and is labelled as such — NOT the L21b metric.

    python docs/dev/transition/pdv-trigger/data/make_bench5_analysis.py
Deliverables: data/bench5_analysis.csv + bench5_theta_tracks.png
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
SUMMARY = PDV / "runs" / "data" / "bench5_summary.csv"
TRAJ = PDV / "runs" / "data" / "bench5_traj"
ELBADRY = HERE / "bench5_elbadry_prediction.csv"
L21B_BAND = (0.90, 0.99)
FIRE = 0.95


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(x for x in fh if not x.lstrip().startswith("#")))


def _fnum(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _bench_of(run_name):
    return run_name.split("__")[0]


def _fa_of(run_name):
    tag = run_name.split("__", 1)[1].replace("_diag", "")
    return 1 if tag == "none" else int(tag[2:])


def theta_cum_prefire(traj_rows):
    """∫θ-weighted: ∫Lloss dt / ∫Lmech dt over the (pre-fire) implicit trajectory, trapezoid.
    NOTE censored at fire — NOT the Lancaster window; a lower bound / early-epoch value only."""
    ts = [_fnum(r["t_now"]) for r in traj_rows]
    Ll = [_fnum(r["Lcool"]) + (_fnum(r["Lleak"]) or 0) for r in traj_rows]  # bubble_Lloss = cool+leak
    Lm = [_fnum(r["Lmech"]) for r in traj_rows]
    pts = [(t, a, b) for t, a, b in zip(ts, Ll, Lm) if t is not None and a is not None and b]
    if len(pts) < 2:
        return None, None, None
    num = den = leak_num = 0.0
    for (t0, l0, m0), (t1, l1, m1) in zip(pts, pts[1:]):
        dt = t1 - t0
        num += 0.5 * (l0 + l1) * dt
        den += 0.5 * (m0 + m1) * dt
    # leak fraction of the loss budget (Rogers & Pittard channel check)
    lk = [(_fnum(r["Lleak"]) or 0.0) for r in traj_rows]
    cool = [(_fnum(r["Lcool"]) or 0.0) for r in traj_rows]
    tot = sum(lk) + sum(cool)
    leak_frac = (sum(lk) / tot) if tot else 0.0
    return (num / den if den else None), pts[-1][0], leak_frac


def main():
    summ = {r["run_name"]: r for r in _read_csv(SUMMARY)}
    eb = {r["bench"]: r for r in _read_csv(ELBADRY)}
    rows = []
    for name in sorted(summ):
        s = summ[name]
        bench, fa = _bench_of(name), _fa_of(name)
        is_diag = name.endswith("_diag")
        tmax = _fnum(s.get("theta_max"))
        traj_path = TRAJ / f"{name}.csv"
        tcum = tfire_traj = leak_frac = None
        if traj_path.exists():
            tcum, tfire_traj, leak_frac = theta_cum_prefire(_read_csv(traj_path))
        ebr = eb.get(bench, {})
        rows.append({
            "run_name": name, "bench": bench, "f_A": fa, "arm": "diag" if is_diag else "prod",
            "fired": s.get("fired_cooling_balance"), "theta_max": f"{tmax:.4f}" if tmax else "",
            "t_fire_Myr": s.get("t_at_theta_max", ""), "t_final_Myr": s.get("t_final", ""),
            "fate": s.get("outcome", ""), "phase_final": s.get("phase_final", ""),
            "theta_cum_prefire": f"{tcum:.4f}" if tcum else "", "leak_frac": f"{leak_frac:.3f}" if leak_frac is not None else "",
            "theta_EB_ldv3": ebr.get("theta_EB_ldv3_Amix3p5", ""), "n_bar_H": ebr.get("n_bar_H", ""),
        })

    out = HERE / "bench5_analysis.csv"
    cols = list(rows[0].keys())
    with out.open("w", newline="") as fh:
        fh.write("# bench5 Phase-5 analysis (PROVISIONAL / IN-CONTAINER, NOT HPC — HPC down 2026-07-12). "
                 "FIRE MAP + theta_max + pre-fire theta_cum from COMPLETED arms only. The Lancaster "
                 "Theta_cum-over-window + dex metric need the DIAGNOSTIC (blowout) arms = HPC-deferred "
                 "(none completed in-container). theta_cum_prefire is the PRE-FIRE trapezoid only, a "
                 "lower bound, NOT the L21b window metric. theta_EB from data/bench5_elbadry_prediction.csv "
                 "(registered, sim-free). L21b band Theta in [0.90,0.99].\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")

    # --- figure: theta(t) tracks for completed arms + theta_EB per bench + fire line ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = {"bench5_m5e5_r2p5": "#c1121f", "bench4_m1e5_r2p5": "#e07a00",
            "bench3_m1e5_r5": "#2a9d3f", "bench2_m1e5_r10": "#1f6feb", "bench1_m5e4_r20": "#6a4c93"}
    plotted = set()
    for name in sorted(summ):
        tp = TRAJ / f"{name}.csv"
        if not tp.exists():
            continue
        tr = _read_csv(tp)
        ts = [_fnum(r["t_now"]) for r in tr]
        th = [_fnum(r["theta"]) for r in tr]
        pts = [(t, h) for t, h in zip(ts, th) if t and h is not None and t > 0]
        if len(pts) < 2:
            continue
        bench = _bench_of(name)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=cmap.get(bench, "#888"),
                lw=1.1, alpha=0.8, label=bench if bench not in plotted else None)
        plotted.add(bench)
    for bench, ebr in eb.items():
        th_eb = _fnum(ebr.get("theta_EB_ldv3_Amix3p5"))
        if th_eb and bench in plotted:
            ax.axhline(th_eb, color=cmap.get(bench, "#888"), ls=":", lw=0.8, alpha=0.6)
    ax.axhline(FIRE, color="k", ls="--", lw=1, label="fire θ=0.95")
    ax.axhspan(*L21B_BAND, color="gray", alpha=0.12, label="L21b Θ band")
    ax.set_xscale("log")
    ax.set_xlabel("t [Myr]  (implicit phase, censored at fire)")
    ax.set_ylabel("θ = L_loss / L_mech")
    ax.set_title("bench5 θ(t) — PROVISIONAL in-container (production arms, censored at fire)\n"
                 "dotted = registered El-Badry θ_EB(λδv=3); diagnostic arms (Θ_cum) HPC-deferred")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    figpath = PDV / "bench5_theta_tracks.png"
    fig.savefig(figpath, dpi=110)
    print(f"wrote {figpath}")

    # --- console fire-map summary ---
    print("\nFIRE MAP (completed arms):")
    benches = ["bench5_m5e5_r2p5", "bench4_m1e5_r2p5", "bench3_m1e5_r5", "bench2_m1e5_r10", "bench1_m5e4_r20"]
    for b in benches:
        fas = sorted({_fa_of(n) for n in summ if _bench_of(n) == b and not n.endswith("_diag")})
        fired = [fa for fa in fas if summ.get(f"{b}__" + ("none" if fa == 1 else f"fa{fa}"), {}).get("fired_cooling_balance") == "True"]
        n = eb.get(b, {}).get("n_bar_H", "?")
        print(f"  {b:20s} n={n:>8}  prod arms done: {fas or '—'}  FIRED: {fired or '—'}")


if __name__ == "__main__":
    main()
