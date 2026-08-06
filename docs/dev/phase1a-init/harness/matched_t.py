#!/usr/bin/env python3
"""Compare two extracted run CSVs at matched simulation time.

CLAUDE.md rule 5: A/B comparisons run in separate processes and are judged at
matched t, never at nearest snapshot (the arms truncate at different times and
their snapshot grids differ). Both arms are linearly interpolated onto the
requested times; times outside either arm's span are reported as out-of-range
rather than extrapolated.

Usage (from repo root):
    python docs/dev/phase1a-init/harness/matched_t.py <stock.csv> <fixed.csv> \
        [--col R2] [--times 3e3,1e4,1e5,1e6] [--last]

`--last` adds the latest time both arms share, which is the "or end of the run
if it terminates earlier" clause of the PLAN.md §4 bar. Times are in YEARS;
the CSVs carry t_now in Myr.
"""
import argparse
import csv

MYR2YR = 1e6


def series(path, col):
    with open(path) as f:
        rows = list(csv.DictReader(r for r in f if not r.startswith('#')))
    return ([float(r['t_now']) * MYR2YR for r in rows], [float(r[col]) for r in rows])


def interp(xs, ys, x):
    if x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stock')
    p.add_argument('fixed')
    p.add_argument('--col', default='R2')
    p.add_argument('--times', default='3e3,1e4,3e4,1e5,3e5,1e6')
    p.add_argument('--last', action='store_true')
    a = p.parse_args()

    ts, vs = series(a.stock, a.col)
    tf, vf = series(a.fixed, a.col)
    times = [float(t) for t in a.times.split(',') if t]
    if a.last:
        times.append(min(ts[-1], tf[-1]))

    print(f"# stock={a.stock} spans {ts[0]:.4g}-{ts[-1]:.6g} yr ({len(ts)} rows)")
    print(f"# fixed={a.fixed} spans {tf[0]:.4g}-{tf[-1]:.6g} yr ({len(tf)} rows)")
    print(f"t_yr,{a.col}_stock,{a.col}_fixed,rel_pct")
    for t in sorted(set(times)):
        s, f = interp(ts, vs, t), interp(tf, vf, t)
        if s is None or f is None:
            print(f"{t:.6g},{'' if s is None else f'{s:.6g}'},"
                  f"{'' if f is None else f'{f:.6g}'},OUT-OF-RANGE")
            continue
        print(f"{t:.6g},{s:.6g},{f:.6g},{100 * (f - s) / s:+.3f}")


if __name__ == '__main__':
    main()
