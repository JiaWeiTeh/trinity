#!/usr/bin/env python3
"""Matched-t comparison of a boosted run vs its `none` baseline (PLAN.md Task B).

Both runs MUST be the same config, produced in SEPARATE processes (run_stamped.py).
This makes the frozen-screen caveat live: boosting cooling lowers Pb -> lowers PdV
-> moves Eb(t)/R2(t)/v2(t) -> moves blowout itself, so the unboosted trajectory is
NOT the state the boosted ODE visits. We therefore report, per run:

  * t_trans   -- authoritative handoff epoch = last implicit-phase t_now
  * blowout_t -- first t with R2 > rCloud  (geometric blowout)
  * ebpeak_t  -- argmax Eb (the energy turnover)
  * end       -- terminal fate (simulationEnd.txt first line)

and across the two: d_t_trans, d_blowout, whether the boosted run handed off via
cooling BEFORE blowout (the thing the boost is supposed to buy), and the matched-t
state divergence (max relative |delta| in R2/v2/Eb over the shared time span).

Usage:
  python compare_live.py <none_run_dir> <boost_run_dir> [--csv out.csv] [--label NAME]
"""
import json
import sys
from pathlib import Path

import numpy as np


def _find(o, k):
    if isinstance(o, dict):
        if k in o and not isinstance(o[k], (dict, list)):
            return o[k]
        for v in o.values():
            r = _find(v, k)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find(v, k)
            if r is not None:
                return r
    return None


def _g(d, k):
    try:
        return float(d.get(k))
    except (TypeError, ValueError):
        return float("nan")


def _end(run_dir):
    p = Path(run_dir) / "simulationEnd.txt"
    if p.exists() and p.read_text().strip():
        return p.read_text().strip().splitlines()[0][:80]
    return "?"


def summarize(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text())
    rCloud = _find(meta, "rCloud")
    rCloud = float(rCloud) if rCloud is not None else float("nan")
    rows = [json.loads(l) for l in (run_dir / "dictionary.jsonl").read_text().splitlines() if l.strip()]
    impl = [d for d in rows if d.get("current_phase") == "implicit"]
    trans = [d for d in rows if d.get("current_phase") == "transition"]
    t = np.array([_g(d, "t_now") for d in impl])
    R2 = np.array([_g(d, "R2") for d in impl])
    v2 = np.array([_g(d, "v2") for d in impl])
    Eb = np.array([_g(d, "Eb") for d in impl])
    t_trans = float(t[-1]) if len(t) else float("nan")
    blowout = float("nan")
    if rCloud == rCloud and len(t):
        cross = np.where(R2 > rCloud)[0]
        if len(cross):
            blowout = float(t[cross[0]])
    ebpeak = float(t[int(np.nanargmax(Eb))]) if len(Eb) and np.isfinite(Eb).any() else float("nan")
    return dict(run=run_dir.name, rCloud=rCloud, n_impl=len(impl), n_trans=len(trans),
                t_trans=t_trans, blowout=blowout, ebpeak=ebpeak, end=_end(run_dir),
                t=t, R2=R2, v2=v2, Eb=Eb)


def matched_div(a, b):
    if len(a["t"]) < 2 or len(b["t"]) < 2:
        return {}
    lo, hi = max(a["t"][0], b["t"][0]), min(a["t"][-1], b["t"][-1])
    mask = (a["t"] >= lo) & (a["t"] <= hi)
    ta = a["t"][mask]
    if len(ta) < 2:
        return {}
    out = {"overlap": (float(lo), float(hi))}
    for key in ("R2", "v2", "Eb"):
        bi = np.interp(ta, b["t"], b[key])
        ai = a[key][mask]
        denom = np.where(np.abs(ai) > 0, np.abs(ai), np.nan)
        out[key] = float(np.nanmax(np.abs(bi - ai) / denom))
    return out


def main():
    args = sys.argv[1:]
    csv = label = None
    if "--csv" in args:
        i = args.index("--csv"); csv = args[i + 1]; args = args[:i] + args[i + 2:]
    if "--label" in args:
        i = args.index("--label"); label = args[i + 1]; args = args[:i] + args[i + 2:]
    none_dir, boost_dir = args[0], args[1]
    a, b = summarize(none_dir), summarize(boost_dir)
    div = matched_div(a, b)
    label = label or b["run"]
    fired_cooling = (b["t_trans"] < b["blowout"] - 1e-9) if b["blowout"] == b["blowout"] else (b["n_trans"] > 0)

    print(f"=== {label}: {b['run']} vs {a['run']} (rCloud={a['rCloud']:.4g} pc) ===")
    for s, tag in ((a, "none "), (b, "boost")):
        print(f"  [{tag}] t_trans={s['t_trans']:.4g}  blowout={s['blowout']:.4g}  "
              f"ebpeak={s['ebpeak']:.4g}  n_impl={s['n_impl']} n_trans={s['n_trans']}  end={s['end']}")
    print(f"  d_t_trans (boost-none) = {b['t_trans'] - a['t_trans']:+.4g} Myr   "
          f"d_blowout = {b['blowout'] - a['blowout']:+.4g} Myr")
    print(f"  boost handed off via cooling BEFORE blowout? {fired_cooling}")
    if div:
        print(f"  matched-t [{div['overlap'][0]:.4g},{div['overlap'][1]:.4g}] Myr  max rel |d|: "
              f"R2={div.get('R2', float('nan')):.3g}  v2={div.get('v2', float('nan')):.3g}  "
              f"Eb={div.get('Eb', float('nan')):.3g}")

    if csv:
        import csv as _csv
        new = not Path(csv).exists()
        Path(csv).parent.mkdir(parents=True, exist_ok=True)
        with open(csv, "a", newline="") as fh:
            w = _csv.writer(fh)
            if new:
                w.writerow(["label", "none_run", "boost_run", "t_trans_none", "t_trans_boost",
                            "d_t_trans", "blowout_none", "blowout_boost", "d_blowout",
                            "ebpeak_none", "ebpeak_boost", "fired_cooling_boost",
                            "maxrel_R2", "maxrel_v2", "maxrel_Eb", "end_none", "end_boost"])
            w.writerow([label, a["run"], b["run"], a["t_trans"], b["t_trans"],
                        b["t_trans"] - a["t_trans"], a["blowout"], b["blowout"],
                        b["blowout"] - a["blowout"], a["ebpeak"], b["ebpeak"], fired_cooling,
                        div.get("R2"), div.get("v2"), div.get("Eb"), a["end"], b["end"]])
        print(f"  wrote {csv}")


if __name__ == "__main__":
    main()
