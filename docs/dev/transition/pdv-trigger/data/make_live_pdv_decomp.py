#!/usr/bin/env python3
"""Live PdV-vs-radiative decomposition of the energy-phase sink, measured on a
FRESH run against CURRENT code (not the frozen-trajectory CSVs).

Reproduce:
  python run.py docs/dev/transition/pdv-trigger/runs/params/<cfg>__none.param
  python docs/dev/transition/pdv-trigger/data/make_live_pdv_decomp.py \
      outputs/pdvlive/<cfg>__none/dictionary.jsonl <cfg>

Emits one summary row per config to live_pdv_decomp.csv:
  PdV/Lmech and L_bubble/Lmech (median over the energy phase, and at the peak-Eb
  row), plus whether Eb crosses 0 (the dead-stop path). PdV = 4*pi*R2^2*Pb*v2 in
  code units, matching energy_phase_ODEs.get_ODE_Edot_pure:280.
"""
import json, sys, csv, os
import numpy as np

def harvest(path, cfg):
    R2=[]; v2=[]; Eb=[]; Pb=[]; Lm=[]; Lb=[]
    with open(path) as f:
        for line in f:
            try: r=json.loads(line)
            except Exception: continue
            if not isinstance(r, dict): continue
            def g(k):
                x=r.get(k); return x if isinstance(x,(int,float)) else np.nan
            R2.append(g('R2')); v2.append(g('v2')); Eb.append(g('Eb'))
            Pb.append(g('Pb')); Lm.append(g('Lmech_total')); Lb.append(g('bubble_LTotal'))
    R2,v2,Eb,Pb,Lm,Lb = map(np.array,(R2,v2,Eb,Pb,Lm,Lb))
    PdV = 4*np.pi*R2**2*Pb*v2
    # restrict to physical energy-phase rows (Eb>0, finite terms)
    m = np.isfinite(PdV) & np.isfinite(Lm) & (Lm>0) & (Eb>0)
    pdv_lm = PdV[m]/Lm[m]; lb_lm = Lb[m]/Lm[m]
    crossed = bool(np.any(np.isfinite(Eb) & (Eb<=0)))
    ipk = int(np.nanargmax(np.where(Eb>0,Eb,-np.inf))) if np.any(Eb>0) else -1
    return {
        'config': cfg, 'n_rows': int(m.sum()),
        'PdV_over_Lmech_med': round(float(np.median(pdv_lm)),4),
        'PdV_over_Lmech_max': round(float(np.max(pdv_lm)),4),
        'Lbub_over_Lmech_med': round(float(np.median(lb_lm)),4),
        'Lbub_over_Lmech_max': round(float(np.max(lb_lm)),4),
        'PdV_over_Lmech_atEbpeak': round(float(PdV[ipk]/Lm[ipk]),4) if ipk>=0 and Lm[ipk]>0 else None,
        'Lbub_over_Lmech_atEbpeak': round(float(Lb[ipk]/Lm[ipk]),4) if ipk>=0 and Lm[ipk]>0 else None,
        'Eb_crosses_zero': crossed,
    }

if __name__=='__main__':
    path,cfg = sys.argv[1], sys.argv[2]
    row = harvest(path, cfg)
    out='docs/dev/transition/pdv-trigger/data/live_pdv_decomp.csv'
    cols=list(row.keys())
    exists=os.path.exists(out)
    rows=[]
    if exists:
        with open(out) as f: rows=[r for r in csv.DictReader(f) if r['config']!=cfg]
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
        w.writerow(row)
    print(row)
