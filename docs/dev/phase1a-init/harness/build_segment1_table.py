#!/usr/bin/env python3
"""Build data/segment1_exit.csv: segment-1 exit state per run vs wind impulse.

Reads the committed per-run CSVs (row 0 = t0 snapshot, row 1 = first
post-segment snapshot) and writes one summary row per run. Run from repo root:

    python docs/dev/phase1a-init/harness/build_segment1_table.py
"""
import csv
import os

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '..', 'data')

RUNS = [
    ('m43_probe.csv', '3e-5', 'on'),
    ('m43_seg1e-5.csv', '1e-5', 'on'),
    ('m43_seg3e-6.csv', '3e-6', 'on'),
    ('m43_noapprox.csv', '3e-5', 'ablated'),
    ('m43_logseg.csv', 'eps=0.1 (log-spaced)', 'ablated'),
    ('gmc_control.csv', '3e-5', 'on'),
    ('gmc_noapprox.csv', '3e-5', 'ablated'),
]


def main():
    out = os.path.join(DATA, 'segment1_exit.csv')
    with open(out, 'w', newline='') as f:
        f.write("# segment-1 exit state per run (row 1 of each per-run CSV) vs cumulative wind impulse\n")
        f.write("# built by harness/build_segment1_table.py from the sibling CSVs; 2026-08-04\n")
        w = csv.writer(f)
        w.writerow(['run', 'SEG_DUR_Myr', 'vd_hack', 't_exit_yr', 'R_exit_pc',
                    'v_exit_kms', 'p_exit', 'wind_impulse', 'p_over_wind'])
        for name, seg, hack in RUNS:
            path = os.path.join(DATA, name)
            if not os.path.exists(path):
                continue
            rows = list(csv.DictReader(l for l in open(path) if not l.startswith('#')))
            r0, r1 = rows[0], rows[1]
            dt = float(r1['t_now']) - float(r0['t_now'])
            imp = float(r0['F_ram_wind']) * dt
            p = float(r1['p_shell'])
            w.writerow([name.replace('.csv', ''), seg, hack,
                        f"{float(r1['t_now'])*1e6:.3f}", f"{float(r1['R2']):.4e}",
                        f"{float(r1['v2_kms']):.1f}", f"{p:.4g}", f"{imp:.4g}",
                        f"{p/imp:.3g}"])
    print('wrote', out)


if __name__ == '__main__':
    main()
