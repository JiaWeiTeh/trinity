#!/usr/bin/env python3
"""Regenerate pdv_regime_budget.csv — the PdV/Lmech-per-regime evidence table.

Pure reads of committed CSVs (no sims):
  ../../cleanroom/data/c0_*_h0.csv            (6 normal configs, hybr h0 baseline)
  ../../../failed-large-clouds/data/budget_*.csv  (fail_repro = heavy 5e9; small_1e6 = control)

PdV = 4*pi*R2^2*v2*Pb  (trinity code units, the same term as get_betadelta.py:434
Edot_from_balance = Lmech - Lloss - 4*pi*R2^2*v2*Pb). First 2 rows dropped as startup
transient. Run from the repo root:  python docs/dev/transition/pdv-trigger/data/make_pdv_regime_table.py
"""
import glob
import numpy as np
import pandas as pd

ROWS = []


def push(config, regime, df, Lmech, Lloss, R2, v2, Pb, Eb, nstart=2):
    d = df.reset_index(drop=True)
    PdV = 4 * np.pi * d[R2] ** 2 * d[v2] * d[Pb]
    lm = d[Lmech]
    ratio = (PdV / lm).replace([np.inf, -np.inf], np.nan)
    net = lm - d[Lloss] - PdV
    s = ratio.iloc[nstart:].dropna()
    eb = d[Eb]
    ebpeak_i = eb.idxmax()
    netpost = net.iloc[nstart:]
    fires = netpost <= 0
    first_fire = netpost[fires].index.min() if fires.any() else None
    ROWS.append(dict(
        config=config, regime=regime,
        PdV_over_Lmech_median=round(float(s.median()), 3),
        PdV_over_Lmech_max=round(float(s.max()), 3),
        Lloss_over_Lmech_median=round(float((d[Lloss] / lm).iloc[nstart:].median()), 4),
        Eb_growth_factor=round(float(eb.max() / eb.iloc[0]), 4) if eb.iloc[0] != 0 else float('nan'),
        Eb_monotonic=bool(ebpeak_i >= len(d) - 2),
        ebpeak_fires_in_cloud=bool(first_fire is not None),
        Eb_peak_row=int(ebpeak_i), Eb_n_rows=len(d) - 1, n=len(d),
    ))


def main():
    for f in sorted(glob.glob('docs/dev/transition/cleanroom/data/c0_*_h0.csv')):
        cfg = f.split('/c0_')[1].rsplit('_h0', 1)[0]
        df = pd.read_csv(f)
        if 'betadelta_converged' in df:
            df = df[df['betadelta_converged'] == True]  # noqa: E712
        df = df[df['Eb'] > 0]
        push(cfg, 'normal', df, 'Lmech_total', 'bubble_Lloss', 'R2', 'v2', 'Pb', 'Eb')

    for f in sorted(glob.glob('docs/dev/failed-large-clouds/data/budget_*.csv')):
        cfg = f.split('budget_')[1].replace('.csv', '')
        regime = 'heavy_5e9' if 'fail' in cfg else 'normal_ctrl'
        df = pd.read_csv(f)
        df['Lloss_tot'] = df['Lcool'] + df.get('Lleak', 0)
        push(cfg, regime, df, 'Lmech', 'Lloss_tot', 'R2', 'v2', 'Pb', 'Eb')

    cols = ['config', 'regime', 'PdV_over_Lmech_median', 'PdV_over_Lmech_max',
            'Lloss_over_Lmech_median', 'Eb_growth_factor', 'Eb_monotonic',
            'ebpeak_fires_in_cloud', 'Eb_peak_row', 'Eb_n_rows', 'n']
    out = pd.DataFrame(ROWS)[cols]
    dst = 'docs/dev/transition/pdv-trigger/data/pdv_regime_budget.csv'
    out.to_csv(dst, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {dst}")


if __name__ == '__main__':
    main()
