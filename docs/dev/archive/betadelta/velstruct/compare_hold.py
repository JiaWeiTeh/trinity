#!/usr/bin/env python
"""Phase 6.1 counterfactual diff: accept-baseline vs reject-and-hold.

Usage: compare_hold.py <baseline.csv> <held.csv>

Quantifies how much rejecting the inflow segments (holding the last physical
structure) moves the macro trajectory. If the deltas are ~0, the inflow is
confirmed immaterial -> close Problem 2.
"""
import csv
import sys

import numpy as np


def load(p):
    rows = list(csv.DictReader(open(p)))
    out = {}
    for k in ('t_now', 'R2', 'v2', 'Eb', 'bubble_dMdt', 'v_struct_nneg',
              'beta_plus_delta'):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, KeyError):
                vals.append(float('nan'))
        out[k] = np.array(vals)
    return out


def main():
    base, held = load(sys.argv[1]), load(sys.argv[2])
    nhold = int(np.count_nonzero(held['v_struct_nneg'] == -1))
    print(f"=== {sys.argv[2].split('/')[-1]} vs baseline ===")
    print(f"  segments: base={len(base['t_now'])} held={len(held['t_now'])}  "
          f"holds triggered={nhold}")

    print("  final-state deltas (at the last common stop_t):")
    for k in ('R2', 'v2', 'Eb'):
        bf, hf = base[k][-1], held[k][-1]
        print(f"    {k:4s}: base={bf:.6g}  held={hf:.6g}  "
              f"delta={(hf - bf) / bf * 100:+.4f}%")

    # max deviation across the overlapping trajectory (interp held onto base t)
    tmin = max(base['t_now'].min(), held['t_now'].min())
    tmax = min(base['t_now'].max(), held['t_now'].max())
    m = (base['t_now'] >= tmin) & (base['t_now'] <= tmax)
    print("  max |delta|/value across the full overlapping trajectory:")
    for k in ('R2', 'v2', 'Eb', 'bubble_dMdt'):
        hi = np.interp(base['t_now'][m], held['t_now'], held[k])
        with np.errstate(invalid='ignore', divide='ignore'):
            dev = np.abs((hi - base[k][m]) / base[k][m])
        print(f"    {k:11s}: {np.nanmax(dev) * 100:.4f}%")


if __name__ == '__main__':
    main()
