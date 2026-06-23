#!/usr/bin/env python3
"""Print a sampled trajectory for one traj CSV (collapse-config evidence) and a
matched-t V0-vs-EBFLOOR diff for the no-op check. Reads the committed
traj/h3_traj_<cfg>_<variant>.csv files.

Usage:
  python h3_traj_sample.py traj/h3_traj_fail_repro_EBFLOOR.csv      # sample one
  python h3_traj_sample.py --noop fail_repro                        # V0 vs EBFLOOR
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ = os.path.join(HERE, "traj")


def load(path):
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.append(r)
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sample_rows(rows, n=12):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    idx = sorted(set(int(i * step) for i in range(n)) | {len(rows) - 1})
    return [rows[i] for i in idx]


def print_traj(path):
    rows = load(path)
    print(f"# {os.path.basename(path)}  ({len(rows)} rows)")
    print(f"  {'t_now':>10} {'phase':>9} {'R2':>9} {'v2':>9} {'Eb':>12} "
          f"{'Pb':>11} {'R1':>9} {'ratio':>7}")
    for r in sample_rows(rows):
        print(f"  {_f(r['t_now']) or 0:10.6f} {r['phase']:>9} "
              f"{_f(r['R2']) or 0:9.4f} {_f(r['v2']) or 0:9.1f} "
              f"{_f(r['Eb']) or 0:12.4e} {_f(r['Pb']) or 0:11.3e} "
              f"{_f(r['R1']) or 0:9.4f} "
              f"{(_f(r['ratio']) if r.get('ratio') not in (None,'') else float('nan')):7.4f}")
    # R2 monotonic? v2 finite?
    R2s = [_f(r['R2']) for r in rows if _f(r['R2']) is not None]
    v2s = [_f(r['v2']) for r in rows if _f(r['v2']) is not None]
    if R2s:
        grows = all(b >= a - 1e-9 for a, b in zip(R2s, R2s[1:]))
        print(f"  R2: {R2s[0]:.4f} -> {R2s[-1]:.4f}  monotonic_nondecreasing={grows}")
    if v2s:
        anyneg = any(v < 0 for v in v2s)
        anynan = any(v != v for v in v2s)
        print(f"  v2: {v2s[0]:.1f} -> {v2s[-1]:.1f}  any_negative={anyneg} any_nan={anynan}")


def noop(cfg):
    v0 = load(os.path.join(TRAJ, f"h3_traj_{cfg}_V0.csv"))
    eb = load(os.path.join(TRAJ, f"h3_traj_{cfg}_EBFLOOR.csv"))
    n = min(len(v0), len(eb))
    print(f"# no-op {cfg}: V0 has {len(v0)} rows, EBFLOOR {len(eb)}; comparing first {n}")
    maxdR2 = maxdEb = maxdv2 = 0.0
    for a, b in zip(v0[:n], eb[:n]):
        for key, acc in (("R2", "R2"), ("Eb", "Eb"), ("v2", "v2")):
            fa, fb = _f(a[key]), _f(b[key])
            if fa is None or fb is None:
                continue
            d = abs(fa - fb)
            if key == "R2":
                maxdR2 = max(maxdR2, d)
            elif key == "Eb":
                maxdEb = max(maxdEb, d / (abs(fa) + 1e-300))
            else:
                maxdv2 = max(maxdv2, d)
    print(f"  max|dR2|={maxdR2:.3e} pc   max rel|dEb|={maxdEb:.3e}   max|dv2|={maxdv2:.3e} km/s")
    print(f"  => {'BIT/VALUE-IDENTICAL (no-op confirmed)' if (maxdR2==0 and maxdEb==0 and maxdv2==0) else ('TRACK-IDENTICAL within fp' if (maxdR2<1e-9 and maxdEb<1e-12) else 'DIFFERS')}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--noop":
        noop(sys.argv[2])
    elif len(sys.argv) >= 2:
        print_traj(sys.argv[1])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
