#!/usr/bin/env python3
"""P22 step 2 — FROZEN reduction of the 72-arm in-container C_f scan.

⚠️ PROVISIONAL / IN-CONTAINER — NOT HPC. INTERIM tier (PLAN §0.12). Never a paper
number; never fills a TBD(HPC); does NOT satisfy/shrink P21 (the ~8000-run Helix
survey remains the sole data SSOT).

This is the maintainer's "reduce the committed dicts with the frozen matcher" step
the README close-out (§11) requests. It parses the 72 gzipped raw dictionary.jsonl
directly (trinity_reader needs scipy, absent in-container) and calls the FROZEN
paper/rosette/matching/ pipeline (observables.py + likelihood.match_run) — the only
quotable matcher. Radii-only default χ² (R2 7±1, rShell 19±2; window 1.5-2.5 Myr).

F-12: both cavity bases reported (7.0 native; 6.2 via an IN-MEMORY OBS override that
never writes observables.py — the file on disk is unchanged). Report both, resolve
neither.

Matched-t discipline: likelihood.match_run interpolates only within each arm's
[t[0], t[-1]] and returns +inf beyond, so nothing is extrapolated past t_final; the
age-censored arms (t_final < 1.5 Myr) get status 'ended_before_window' and are kept
visible but excluded from every minimum.

Command:
  python docs/dev/rosette-cf/figs/reduce_cf_scan_frozen.py
Outputs (committed home, 💾 rule; plots/ is gitignored so we write to data/):
  docs/dev/rosette-cf/data/match_interim_cf_PISM1e5_frozen_<date>.csv        (per-arm)
  docs/dev/rosette-cf/data/match_interim_cf_PISM1e5_frozen_<date>_cells.csv  (per-cell)
"""
import gzip, json, re, sys, glob, os, csv, datetime
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
MATCHING = os.path.join(ROOT, 'paper', 'rosette', 'matching')
DICTS = os.path.join(ROOT, 'docs', 'dev', 'rosette-cf', 'data', 'cf_scan_PISM1e5_dicts')
OUTDIR = os.path.join(ROOT, 'docs', 'dev', 'rosette-cf', 'data')
DATE = datetime.date.today().isoformat()

sys.path.insert(0, MATCHING)
import observables  # noqa: E402
from likelihood import match_run  # noqa: E402
from observables import DEFAULT_TERMS  # radii-only  # noqa: E402

CF_GRID = [0.70, 0.85, 1.0]
BASES = {'7': 7.0, '62': 6.2}
MATCHABLE = {'ok', 'ok_censored'}


def parse_axes(run_name):
    mCloud = float(run_name.split('_')[0])
    sfe = int(re.search(r'_sfe(\d+)', run_name).group(1)) / 100.0
    nCore = float(re.search(r'_n([0-9]+(?:e[0-9]+)?)_', run_name).group(1))
    cf = float(re.search(r'coverFraction([0-9p]+)', run_name).group(1).replace('p', '.'))
    fmix = float(re.search(r'coolingBoostFmix([0-9p]+)', run_name).group(1).replace('p', '.'))
    phii = 'yesPHII' in run_name
    return dict(mCloud=mCloud, sfe=sfe, nCore=nCore, coverFraction=cf,
                cooling_boost_fmix=fmix, include_PHII=phii)


def load_arrays(path):
    t, R2, rSh, v2, isC, isD = [], [], [], [], [], []
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            t.append(r['t_now']); R2.append(r['R2']); rSh.append(r['rShell'])
            v2.append(r['v2']); isC.append(bool(r.get('isCollapse', False)))
            isD.append(bool(r.get('isDissolved', False)))
    return (np.array(t), np.array(R2), np.array(rSh), np.array(v2),
            np.array(isC, bool), np.array(isD, bool))


def match_both_bases(t, R2, rSh, v2, isC, isD):
    out = {}
    orig = observables.OBS['R_bubble']['value']
    try:
        for tag, base in BASES.items():
            observables.OBS['R_bubble']['value'] = base   # in-memory only; file untouched
            r = match_run(t, R2, rSh, v2=v2, isCollapse=isC, isDissolved=isD,
                          terms=DEFAULT_TERMS)
            out[tag] = dict(r, base=base)
    finally:
        observables.OBS['R_bubble']['value'] = orig       # restore native 7.0
    return out


def main():
    files = sorted(glob.glob(os.path.join(DICTS, '*.jsonl.gz')))
    assert len(files) == 72, f"expected 72 dicts, found {len(files)}"
    arms = []
    for f in files:
        run = os.path.basename(f)[:-len('.jsonl.gz')]
        ax = parse_axes(run)
        t, R2, rSh, v2, isC, isD = load_arrays(f)
        m = match_both_bases(t, R2, rSh, v2, isC, isD)
        age_censored = (t[-1] < observables.AGE_WINDOW_MYR[0]) or \
                       (m['7']['status'] == 'ended_before_window')
        arms.append(dict(run_name=run, t=t, R2=R2, rSh=rSh, t_final=float(t[-1]),
                         n_snap=len(t), R2_final=float(R2[-1]), rShell_final=float(rSh[-1]),
                         age_censored=bool(age_censored), m=m, **ax))
        print(f"  {run:70s} tf={t[-1]:.2f} st7={m['7']['status']:18s} "
              f"chi2_7={m['7']['chi2_min']:.1f} R2@best={m['7'].get('R2_at_best',float('nan')):.1f}")

    # ---- per-arm CSV ----
    stamp = [f"# ⚠️ PROVISIONAL / IN-CONTAINER — NOT HPC. INTERIM tier (PLAN §0.12).",
             f"# FROZEN reduction of the 72-arm in-container C_f scan (docs/dev/rosette-cf).",
             f"# matcher: paper/rosette/matching/ (observables.py + likelihood.match_run), radii-only.",
             f"# both cavity bases: _7 (7.0±1.0 pc, frozen native) and _62 (6.2 pc, in-memory override; F-12).",
             f"# matched-t: no extrapolation past t_final; age-censored (t_final<1.5) excluded from minima, kept visible.",
             f"# generated {datetime.datetime.utcnow().isoformat()}Z | cmd: python docs/dev/rosette-cf/figs/reduce_cf_scan_frozen.py",
             f"# NEVER a paper number; never fills TBD(HPC); does not satisfy P21."]
    per_arm = os.path.join(OUTDIR, f'match_interim_cf_PISM1e5_frozen_{DATE}.csv')
    cols = ['run_name', 'mCloud', 'sfe', 'nCore', 'coverFraction', 'cooling_boost_fmix',
            'include_PHII', 't_final', 'n_snap', 'R2_final', 'rShell_final', 'age_censored',
            'status_7', 'chi2_min_7', 't_best_7', 'R2_at_best_7', 'rShell_at_best_7', 'over7',
            'lnL_marg_7', 'v2_kms_at_best_7', 'vShell_kms_at_best_7',
            'status_62', 'chi2_min_62', 't_best_62', 'R2_at_best_62', 'over62', 'lnL_marg_62']
    with open(per_arm, 'w', newline='') as fh:
        for s in stamp:
            fh.write(s + "\n")
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for a in arms:
            m7, m62 = a['m']['7'], a['m']['62']
            r2b7 = m7.get('R2_at_best'); r2b62 = m62.get('R2_at_best')
            w.writerow(dict(
                run_name=a['run_name'], mCloud=a['mCloud'], sfe=a['sfe'], nCore=a['nCore'],
                coverFraction=a['coverFraction'], cooling_boost_fmix=a['cooling_boost_fmix'],
                include_PHII=a['include_PHII'], t_final=round(a['t_final'], 4), n_snap=a['n_snap'],
                R2_final=round(a['R2_final'], 3), rShell_final=round(a['rShell_final'], 3),
                age_censored=a['age_censored'],
                status_7=m7['status'], chi2_min_7=_r(m7['chi2_min']), t_best_7=_r(m7.get('t_best')),
                R2_at_best_7=_r(r2b7), rShell_at_best_7=_r(m7.get('rShell_at_best')),
                over7=_r(None if r2b7 is None or not np.isfinite(r2b7) else r2b7 - 7.0),
                lnL_marg_7=_r(m7.get('lnL_marg')), v2_kms_at_best_7=_r(m7.get('v2_kms_at_best')),
                vShell_kms_at_best_7=_r(m7.get('vShell_kms_at_best')),
                status_62=m62['status'], chi2_min_62=_r(m62['chi2_min']), t_best_62=_r(m62.get('t_best')),
                R2_at_best_62=_r(r2b62),
                over62=_r(None if r2b62 is None or not np.isfinite(r2b62) else r2b62 - 6.2),
                lnL_marg_62=_r(m62.get('lnL_marg'))))

    # ---- per-cell CSV ----
    cells = {}
    for a in arms:
        key = (a['mCloud'], a['sfe'], a['nCore'], a['cooling_boost_fmix'], a['include_PHII'])
        cells.setdefault(key, {})[a['coverFraction']] = a
    cell_rows = []
    for key, byc in sorted(cells.items()):
        mCloud, sfe, nCore, fmix, phii = key
        present = [byc[c] for c in CF_GRID if c in byc]
        t_match = min(min(2.5, a['t_final']) for a in present)   # matched-t, never > any t_final
        row = dict(mCloud=mCloud, sfe=sfe, nCore=nCore, cooling_boost_fmix=fmix,
                   include_PHII=phii, t_match=round(t_match, 4), n_arms=len(present))
        for tag, base in BASES.items():
            chi = {c: byc[c]['m'][tag]['chi2_min'] for c in CF_GRID
                   if c in byc and byc[c]['m'][tag]['status'] in MATCHABLE}
            full3 = len(chi) == 3
            best_cf = min(chi, key=chi.get) if chi else None
            edge_min = (best_cf in (CF_GRID[0], CF_GRID[-1])) if best_cf is not None else None
            row[f'n_matchable_{tag}'] = len(chi)
            row[f'chi2_grid_{tag}'] = ';'.join(_r(chi.get(c)) if c in chi else 'NA' for c in CF_GRID)
            row[f'best_cf_{tag}'] = best_cf
            row[f'edge_min_{tag}'] = edge_min
            row[f'full3_{tag}'] = full3
        # R2 / overshoot at t_match per Cf (interp within trajectory -> no extrapolation)
        r2tm = {}
        for c in CF_GRID:
            if c in byc:
                a = byc[c]
                r2tm[c] = float(np.interp(t_match, a['t'], a['R2']))
        row['R2_tmatch_grid'] = ';'.join(_r(r2tm.get(c)) if c in r2tm else 'NA' for c in CF_GRID)
        row['over7_tmatch_grid'] = ';'.join(_r(r2tm[c] - 7.0) if c in r2tm else 'NA' for c in CF_GRID)
        row['over62_tmatch_grid'] = ';'.join(_r(r2tm[c] - 6.2) if c in r2tm else 'NA' for c in CF_GRID)
        row['sealed_over7'] = _r(r2tm[1.0] - 7.0) if 1.0 in r2tm else 'NA'
        row['sealed_over62'] = _r(r2tm[1.0] - 6.2) if 1.0 in r2tm else 'NA'
        cell_rows.append(row)

    per_cell = os.path.join(OUTDIR, f'match_interim_cf_PISM1e5_frozen_{DATE}_cells.csv')
    ccols = ['mCloud', 'sfe', 'nCore', 'cooling_boost_fmix', 'include_PHII', 't_match', 'n_arms',
             'n_matchable_7', 'chi2_grid_7', 'best_cf_7', 'edge_min_7', 'full3_7',
             'n_matchable_62', 'chi2_grid_62', 'best_cf_62', 'edge_min_62', 'full3_62',
             'R2_tmatch_grid', 'over7_tmatch_grid', 'over62_tmatch_grid', 'sealed_over7', 'sealed_over62']
    with open(per_cell, 'w', newline='') as fh:
        for s in stamp:
            fh.write(s + "\n")
        fh.write("# cf_grid order for all ';'-joined columns: 0.70;0.85;1.0\n")
        w = csv.DictWriter(fh, fieldnames=ccols); w.writeheader()
        for r in cell_rows:
            w.writerow(r)

    # ---- console summary ----
    n_match = sum(1 for a in arms if a['m']['7']['status'] in MATCHABLE)
    n_cens = sum(1 for a in arms if a['age_censored'])
    full3_cells = sum(1 for r in cell_rows if r['full3_7'])
    edge_cells = sum(1 for r in cell_rows if r['full3_7'] and r['edge_min_7'])
    print(f"\n== FROZEN reduction summary ==")
    print(f"arms: 72 | matchable(7): {n_match} | age-censored: {n_cens}")
    print(f"full-3-point cells: {full3_cells}/24 | of those edge-min(7): {edge_cells}")
    sealed = [(a['nCore'], a['include_PHII'], a['cooling_boost_fmix'], a['m']['7'].get('R2_at_best'))
              for a in arms if a['coverFraction'] == 1.0 and a['m']['7']['status'] in MATCHABLE]
    ov = [r2 - 7.0 for _, _, _, r2 in sealed if r2 and np.isfinite(r2)]
    if ov:
        print(f"sealed(Cf=1) over7 at own best-t: min {min(ov):.1f}  max {max(ov):.1f} pc "
              f"(n={len(ov)} matchable sealed arms)")
    print(f"\nwrote:\n  {per_arm}\n  {per_cell}")


def _r(x, nd=3):
    if x is None:
        return 'NA'
    try:
        if not np.isfinite(x):
            return 'inf' if x > 0 else '-inf'
    except (TypeError, ValueError):
        return str(x)
    return str(round(float(x), nd))


if __name__ == '__main__':
    main()
