#!/usr/bin/env python3
"""Extract a compact per-snapshot CSV from a TRINITY dictionary.jsonl.

Usage (from repo root):
    python docs/dev/phase1a-init/harness/extract_csv.py <run_output_dir> <out.csv> "<provenance>"

Writes one row per snapshot with the columns needed by the phase-1a
investigation figures/audits. Values stay in TRINITY AU (pc, Myr, Msun)
except v2_kms. Refuses non-monotonic time series (a restarted/mixed run).
"""
import csv
import json
import sys

COLS = ['t_now', 'R2', 'v2', 'shell_mass', 'Eb', 'Pb', 'P_HII', 'P_drive',
        'F_grav', 'F_rad', 'F_ram_wind', 'Lmech_total', 'current_phase']


def main():
    run_dir, out_csv, provenance = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [json.loads(l) for l in open(f'{run_dir}/dictionary.jsonl') if l.strip()]
    t_prev = -1.0
    for r in rows:
        if r['t_now'] < t_prev:
            sys.exit(f"non-monotonic t_now in {run_dir}: mixed/restarted run, refusing")
        t_prev = r['t_now']
    with open(out_csv, 'w', newline='') as f:
        f.write(f"# {provenance}\n")
        f.write("# columns in TRINITY AU (Msun, pc, Myr) except v2_kms; p_shell = shell_mass*v2\n")
        w = csv.writer(f)
        w.writerow(COLS + ['v2_kms', 'p_shell'])
        for r in rows:
            vals = [r.get(c, '') for c in COLS]
            w.writerow(vals + [r['v2'] * 0.977792, r['shell_mass'] * r['v2']])
    print(f"{out_csv}: {len(rows)} rows")


if __name__ == '__main__':
    main()
