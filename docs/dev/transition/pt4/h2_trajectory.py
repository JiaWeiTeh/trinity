#!/usr/bin/env python3
"""
Dump the R2/rCloud vs cooling-ratio vs Lloss trajectory around the rCloud
crossing for a couple of representative configs, to show whether Lloss keeps
falling after the shell leaves the cloud (deep in nISM) and whether the
cooling ratio ever re-approaches 0.05 post-crossing.

Read-only of committed h0 CSVs. No sims.
Run: python docs/dev/transition/pt4/h2_trajectory.py
"""
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "cleanroom", "data"))

RCLOUD = {"simple_cluster": 1.69, "large_diffuse_lowsfe": 88.05,
          "small_dense_highsfe": 0.3255}


def load(name):
    rows = []
    with open(os.path.join(DATA, f"c0_{name}_h0.csv")) as f:
        for row in csv.DictReader(f):
            def g(k):
                try:
                    return float(row.get(k, ""))
                except (ValueError, TypeError):
                    return np.nan
            rows.append((g("t_now"), g("R2"), g("bubble_Lgain"),
                         g("bubble_Lloss"), g("Eb"), g("T0"), g("Pb"),
                         row.get("phase", "")))
    rows.sort(key=lambda r: r[0] if np.isfinite(r[0]) else 0.0)
    return rows


for name, rc in RCLOUD.items():
    rows = load(name)
    print("=" * 78)
    print(f"{name}   rCloud={rc} pc")
    print(f"{'t[Myr]':>10s} {'R2[pc]':>10s} {'R2/rC':>7s} {'ratio':>8s} "
          f"{'Lgain':>10s} {'Lloss':>10s} {'Lloss/Lg':>8s} {'T0[K]':>9s} {'phase':>9s}")
    # print a thinned sample: every Nth row plus rows near the crossing
    cross_i = next((i for i, r in enumerate(rows) if np.isfinite(r[1]) and r[1] >= rc), None)
    n = len(rows)
    keep = set(range(0, n, max(1, n // 18)))
    if cross_i is not None:
        keep.update(range(max(0, cross_i - 2), min(n, cross_i + 3)))
    for i in sorted(keep):
        t, R2, Lg, Ll, Eb, T0, Pb, ph = rows[i]
        ratio = (Lg - Ll) / Lg if (np.isfinite(Lg) and Lg > 0) else np.nan
        llg = Ll / Lg if (np.isfinite(Lg) and Lg > 0) else np.nan
        mark = "  <-- crosses rCloud" if i == cross_i else ""
        print(f"{t:10.4g} {R2:10.4g} {R2/rc:7.3g} {ratio:8.4f} "
              f"{Lg:10.3g} {Ll:10.3g} {llg:8.4f} {T0:9.3g} {ph:>9s}{mark}")
