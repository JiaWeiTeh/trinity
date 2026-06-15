#!/usr/bin/env python3
"""P0 harvest: both transition clocks + candidate trigger firing epochs.

Reads a finished run's dictionary.jsonl (+ metadata.json for run constants) and,
per implicit-phase segment, evaluates every candidate transition trigger on the
SAME trajectory so we can see where they diverge. Pure offline read — no
production change. See docs/dev/TRANSITION_TRIGGER_PLAN.md (P0).

Usage: python scratch/transition/harvest.py outputs/<model>/ [--csv out.csv]

Candidates (plan F0–F4):
  F0 instantaneous rate-ratio (current): (Lgain-Lloss)/Lgain < eps   [eps=0.05]
  F1 cumulative energy:  ∫Lloss dt / ∫Lgain dt > 1-eta   [eta in 0.20..0.40]
  F2 timescale:          t_cool/t_dyn < k,  t_cool=Eb/Lloss, t_dyn=R2/v2  [k=1,2,3]
  F4 blowout:            R2 > rCloud
Reference physical event: Eb-peak (argmax Eb) — "bubble stops gaining energy".
F3 (force ratio) components are logged raw (F_ram=4πR²Pb vs F_rad,F_HII,pdot)
   for later; the exact 'surviving-force' set is pinned in P-sens, not here.

Clocks:
  A = t_trans   : first/last implicit time (when the run leaves the energy phase)
  B = 1c length : duration of the transition phase (Eb drain to floor / stop_t)
"""
import json
import sys
from pathlib import Path

EPS = 0.05
ETAS = (0.20, 0.25, 0.30, 0.40)
KS = (1.0, 2.0, 3.0)


def _find_key(obj, name):
    """Recursively search a nested dict/list for the first value at key `name`."""
    if isinstance(obj, dict):
        if name in obj and not isinstance(obj[name], (dict, list)):
            return obj[name]
        for v in obj.values():
            r = _find_key(v, name)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, name)
            if r is not None:
                return r
    return None


def load(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())
    rCloud = _find_key(meta, "rCloud")
    rows = []
    with (run_dir / "dictionary.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows, rCloud


def g(d, k):
    v = d.get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def first_cross(ts, vals, op, thr):
    """First time `op(val, thr)` holds (op: '<' or '>'); None if never."""
    for t, v in zip(ts, vals):
        if v == v and ((op == "<" and v < thr) or (op == ">" and v > thr)):
            return t
    return None


def harvest(run_dir, csv_path=None):
    rows, rCloud = load(run_dir)
    impl = [d for d in rows if d.get("current_phase") == "implicit"]
    trans = [d for d in rows if d.get("current_phase") == "transition"]
    if not impl:
        print(f"!! no implicit-phase rows in {run_dir}")
        return

    t = [g(d, "t_now") for d in impl]
    R2 = [g(d, "R2") for d in impl]
    v2 = [g(d, "v2") for d in impl]
    Eb = [g(d, "Eb") for d in impl]
    Pb = [g(d, "Pb") for d in impl]
    beta = [g(d, "cool_beta") for d in impl]
    delta = [g(d, "cool_delta") for d in impl]
    Lgain = [g(d, "bubble_Lgain") for d in impl]
    Lloss = [g(d, "bubble_Lloss") for d in impl]

    # F0 instantaneous ratio
    ratio = [(lg - ll) / lg if lg > 0 else float("nan") for lg, ll in zip(Lgain, Lloss)]
    # F1 cumulative (left-rectangle: segment k spans [t_k, t_{k+1}], uses value_k)
    cumL, cumG, fcum = [], [], []
    aL = aG = 0.0
    for k in range(len(impl)):
        dt = (t[k + 1] - t[k]) if k + 1 < len(impl) else 0.0
        aL += Lloss[k] * dt
        aG += Lgain[k] * dt
        cumL.append(aL)
        cumG.append(aG)
        fcum.append(aL / aG if aG > 0 else float("nan"))
    # F2 timescale
    t_cool = [eb / ll if ll > 0 else float("inf") for eb, ll in zip(Eb, Lloss)]
    t_dyn = [r / vv if vv > 0 else float("inf") for r, vv in zip(R2, v2)]
    f2 = [tc / td if td > 0 else float("nan") for tc, td in zip(t_cool, t_dyn)]
    # F4 blowout
    r_over_rc = [r / rCloud if rCloud else float("nan") for r in R2]

    # firing epochs
    fire = {
        "F0 inst<0.05": first_cross(t, ratio, "<", EPS),
        **{f"F1 cum>1-{e}": first_cross(t, fcum, ">", 1 - e) for e in ETAS},
        **{f"F2 tc/td<{int(k)}": first_cross(t, f2, "<", k) for k in KS},
        "F4 R2>rCloud": first_cross(t, R2, ">", rCloud) if rCloud else None,
    }
    # reference: Eb peak
    ipk = max(range(len(Eb)), key=lambda i: Eb[i] if Eb[i] == Eb[i] else -1e99)
    t_peak = t[ipk]
    peak_interior = 0 < ipk < len(Eb) - 1

    # clocks
    tA0, tA1 = t[0], t[-1]
    if trans:
        tt = [g(d, "t_now") for d in trans]
        clockB = (min(tt), max(tt), max(tt) - min(tt))
    else:
        clockB = None

    name = Path(run_dir).name
    print(f"\n=== {name}  (implicit segs={len(impl)}, transition segs={len(trans)}) ===")
    print(f"  rCloud = {rCloud:.4g} pc" if rCloud else "  rCloud = ?")
    print(f"  Clock A (implicit span / t_trans): {tA0:.4g} -> {tA1:.4g} Myr  (len {tA1-tA0:.4g})")
    if clockB:
        print(f"  Clock B (1c transition phase):     {clockB[0]:.4g} -> {clockB[1]:.4g} Myr  "
              f"(len {clockB[2]:.4g})")
    else:
        print("  Clock B: no transition phase (energy-driven to stop_t / stall)")
    print(f"  Eb-peak (reference): t={t_peak:.4g} Myr  "
          f"({'INTERIOR' if peak_interior else 'at boundary -> still rising / no peak'})")
    print(f"  ratio_F0 range: [{min(ratio):.3f}, {max(ratio):.3f}]   "
          f"frac_cum end: {fcum[-1]:.3f}   t_cool/t_dyn end: {f2[-1]:.3g}")
    print("  candidate firing epochs (Myr; '—' = never):")
    for k, v in fire.items():
        print(f"    {k:16s}: {v:.4g}" if v is not None else f"    {k:16s}: —")

    if csv_path:
        import csv as _csv
        cols = ["t_now", "current_phase", "R2", "v2", "Eb", "Pb", "cool_beta",
                "cool_delta", "beta_plus_delta", "Lgain", "Lloss", "ratio_F0",
                "cum_Lloss", "cum_Lgain", "frac_cum", "t_cool", "t_dyn",
                "F2_tcool_tdyn", "R2_over_rCloud", "F_ram", "F_rad", "F_HII",
                "pdot_total"]
        with open(csv_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(cols)
            for k in range(len(impl)):
                w.writerow([t[k], "implicit", R2[k], v2[k], Eb[k], Pb[k], beta[k],
                            delta[k], beta[k] + delta[k], Lgain[k], Lloss[k],
                            ratio[k], cumL[k], cumG[k], fcum[k], t_cool[k],
                            t_dyn[k], f2[k], r_over_rc[k], g(impl[k], "F_ram"),
                            g(impl[k], "F_rad"), g(impl[k], "F_HII"),
                            g(impl[k], "pdot_total")])
        print(f"  wrote {csv_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    csv = None
    if "--csv" in args:
        i = args.index("--csv")
        csv = args[i + 1]
        args = args[:i] + args[i + 2:]
    harvest(args[0], csv)
