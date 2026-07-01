#!/usr/bin/env python3
"""Live PdV-vs-radiative decomposition of the energy-phase sink, measured on a
FRESH run against CURRENT code (not the frozen-trajectory CSVs).

Reproduce:
  python run.py <path/to/cfg>.param                     # writes outputs/.../dictionary.jsonl
  python docs/dev/transition/pdv-trigger/data/make_live_pdv_decomp.py \
      outputs/pdvlive/<cfg>/dictionary.jsonl <label>

Emits one summary row per config to live_pdv_decomp.csv. PdV = 4*pi*R2^2*Pb*v2 in
code units, matching energy_phase_ODEs.get_ODE_Edot_pure:280. Key columns:
  PdV_over_Lmech_* / Lbub_over_Lmech_*  -- the sink decomposition (median + at Eb-peak)
  final_phase / reached_momentum        -- did the run hand off to the momentum phase?
  dead_stop  -- Eb went STRICTLY negative AND the run never reached momentum, i.e. the
                ENERGY_COLLAPSED dead-stop (the bug). Distinct from the clean transition
                path where Eb floors at 0 on the way into the momentum phase.
"""
import json, sys, csv, os
import numpy as np

def harvest(path, cfg):
    raw = []
    with open(path) as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if isinstance(r, dict): raw.append(r)
    def col(k):
        return np.array([r.get(k) if isinstance(r.get(k), (int, float)) else np.nan for r in raw])
    R2, v2, Eb, Pb, Lm, Lb = (col(k) for k in ('R2','v2','Eb','Pb','Lmech_total','bubble_LTotal'))
    phases = [str(r.get('current_phase')) for r in raw]
    PdV = 4*np.pi*R2**2*Pb*v2
    m = np.isfinite(PdV) & np.isfinite(Lm) & (Lm > 0) & (Eb > 0)   # physical energy-phase rows
    pdv_lm, lb_lm = PdV[m]/Lm[m], Lb[m]/Lm[m]
    ipk = int(np.nanargmax(np.where(Eb > 0, Eb, -np.inf))) if np.any(Eb > 0) else -1
    reached_momentum = any(p == 'momentum' for p in phases)
    dead_stop = bool(np.nanmin(Eb) < 0) and not reached_momentum   # ENERGY_COLLAPSED (the bug)
    return {
        'config': cfg, 'n_rows': int(m.sum()),
        'PdV_over_Lmech_med': round(float(np.median(pdv_lm)), 4),
        'Lbub_over_Lmech_med': round(float(np.median(lb_lm)), 4),
        'PdV_over_Lmech_atEbpeak': round(float(PdV[ipk]/Lm[ipk]), 4) if ipk >= 0 and Lm[ipk] > 0 else None,
        'Lbub_over_Lmech_atEbpeak': round(float(Lb[ipk]/Lm[ipk]), 4) if ipk >= 0 and Lm[ipk] > 0 else None,
        'Eb_min': f'{float(np.nanmin(Eb)):.3e}',
        'final_phase': phases[-1] if phases else None,
        'reached_momentum': reached_momentum,
        'dead_stop': dead_stop,
    }

if __name__ == '__main__':
    path, cfg = sys.argv[1], sys.argv[2]
    row = harvest(path, cfg)
    out = 'docs/dev/transition/pdv-trigger/data/live_pdv_decomp.csv'
    cols = list(row.keys())
    prev = []
    if os.path.exists(out):
        with open(out) as f:
            prev = [r for r in csv.DictReader(f) if r['config'] != cfg and set(r) == set(cols)]
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in prev: w.writerow(r)
        w.writerow(row)
    print(row)
