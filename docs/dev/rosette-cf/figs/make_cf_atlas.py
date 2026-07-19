#!/usr/bin/env python3
"""P22 step 4 — INTERIM ATLAS (A1-A6) for the 72-arm in-container C_f scan.

⚠️ PROVISIONAL / IN-CONTAINER — NOT HPC. INTERIM tier (PLAN §0.12). Never a paper
number; never fills a TBD(HPC); does NOT satisfy/shrink P21 (the ~8000-run Helix
survey remains the sole data SSOT).

Reads the FROZEN-reduced CSVs (match_interim_cf_PISM1e5_frozen_<date>{,_cells}.csv,
produced by reduce_cf_scan_frozen.py) + the raw gzipped dicts. Both cavity bases
(7.0 native and 6.2, F-12 — report both, resolve neither). Matched-t: nothing is
plotted past an arm's t_final; the 19 age-censored arms are shown but drawn dimmed
and excluded from every minimum.

A1/A2 are INTERIM STAND-INS for figures/paper_Cf.py, which cannot run in-container
(its trinity_reader import needs scipy, absent here across sessions). The R2/rShell/
v2/t arrays are the SAME raw dict arrays paper_Cf.py would read; only the plotting
wrapper differs. The P21 pass, in a full env, runs the real paper_Cf.py /
paper_Rosette.py as a DATA-SWAP (these panels are the layout dry-run for G-F3/G-F6).

Command: python docs/dev/rosette-cf/figs/make_cf_atlas.py
Outputs: docs/dev/rosette-cf/figs/A{1..6}_*.png
"""
import gzip, json, re, os, glob, csv, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
DICTS = os.path.join(DATA, 'cf_scan_PISM1e5_dicts')
DATE = '2026-07-14'
ARM_CSV = os.path.join(DATA, f'match_interim_cf_PISM1e5_frozen_{DATE}.csv')
CELL_CSV = os.path.join(DATA, f'match_interim_cf_PISM1e5_frozen_{DATE}_cells.csv')

CF_GRID = [0.70, 0.85, 1.0]
NCORE = [50.0, 100.0, 500.0]
CF_COLORS = {1.0: '#2166ac', 0.85: '#f4a582', 0.70: '#b2182b'}   # sealed->leaky
R7, R7E, R62 = 7.0, 1.0, 6.2
RSHELL, RSHELLE = 19.0, 2.0
AGEWIN = (1.5, 2.5)
WM = ("PROVISIONAL / IN-CONTAINER — INTERIM tier, NOT HPC (PLAN §0.12). "
      "Never a paper number; does not satisfy P21. Frozen matcher, both cavity bases (7.0 & 6.2 pc).")


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def load_rows(path):
    return [r for r in csv.DictReader([l for l in open(path) if not l.startswith('#')])]


def cf_of(name):
    return float(re.search(r'coverFraction([0-9p]+)', name).group(1).replace('p', '.'))


def load_traj(run_name):
    f = os.path.join(DICTS, run_name + '.jsonl.gz')
    t, R2, rSh, v2 = [], [], [], []
    for line in gzip.open(f, 'rt'):
        if not line.strip():
            continue
        d = json.loads(line)
        t.append(d['t_now']); R2.append(d['R2']); rSh.append(d['rShell']); v2.append(d['v2'])
    return np.array(t), np.array(R2), np.array(rSh), np.array(v2)


def watermark(fig):
    fig.text(0.5, 0.008, WM, ha='center', va='bottom', fontsize=7, color='#b2182b', wrap=True)


def obs_bands_R2(ax):
    ax.axhspan(R7 - R7E, R7 + R7E, color='#4daf4a', alpha=0.15, zorder=0)
    ax.axhline(R7, color='#4daf4a', lw=1.0, ls='-', zorder=1)
    ax.axhline(R62, color='#984ea3', lw=1.0, ls='--', zorder=1)


def age_band(ax):
    ax.axvspan(*AGEWIN, color='0.85', alpha=0.5, zorder=0)


ARMS = load_rows(ARM_CSV)
CELLS = load_rows(CELL_CSV)
BY_RUN = {r['run_name']: r for r in ARMS}


# ---------------------------------------------------------------- A1 trajectories
def A1_trajectories():
    """Decreasing-C_f R2(t) & rShell(t) for the best mass/PHII combo (1e4, PHII off),
    grid nCore x fmix; obs bands + age window; age-censored curves dimmed."""
    for mass, phii_tag, tag in [('10000.0', 'noPHII', '1e4_noPHII'), ('100000.0', 'yesPHII', '1e5_yesPHII')]:
        fig, axes = plt.subplots(3, 2, figsize=(9, 10), sharex=True)
        for i, nc in enumerate(NCORE):
            for j, fmix in enumerate([1.0, 4.0]):
                ax = axes[i][j]
                age_band(ax); obs_bands_R2(ax)
                for r in ARMS:
                    if fnum(r['mCloud']) != float(mass) or fnum(r['nCore']) != nc:
                        continue
                    if fnum(r['cooling_boost_fmix']) != fmix:
                        continue
                    if (r['include_PHII'] == 'True') != (phii_tag == 'yesPHII'):
                        continue
                    cf = fnum(r['coverFraction'])
                    t, R2, rSh, v2 = load_traj(r['run_name'])
                    dim = r['age_censored'] == 'True'
                    ax.plot(t, R2, color=CF_COLORS[cf], lw=1.6, alpha=0.35 if dim else 1.0,
                            ls=':' if dim else '-',
                            label=f"Cf={cf:g}" + (" (age-cens.)" if dim else ""))
                ax.set_title(f"nCore={nc:g}, fmix={fmix:g}", fontsize=9)
                ax.set_ylim(0, 60)
                if i == 2:
                    ax.set_xlabel("t [Myr]")
                if j == 0:
                    ax.set_ylabel(r"$R_2$ [pc]")
                ax.legend(fontsize=6, loc='upper left')
        fig.suptitle(f"A1 — decreasing-$C_f$ cavity trajectories $R_2(t)$  [{tag}, PL0, PISM=1e5]\n"
                     f"green band 7$\\pm$1 pc, purple dashed 6.2 pc, grey = age window "
                     f"[interim stand-in for paper_Cf.plot_cf_trajectory; scipy absent]",
                     fontsize=10)
        watermark(fig)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        out = os.path.join(HERE, f'A1_cf_trajectories_{tag}.png')
        fig.savefig(out, dpi=130); plt.close(fig)
        print("wrote", out)


# ---------------------------------------------------------------- A2 constraint
def A2_constraint():
    """value-at-age vs C_f: R2 at profiled best-t (within [1.5,2.5]) vs C_f, per nCore,
    against both obs bases. 'which C_f reproduces the cavity'. Both mass/PHII overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    styles = {('10000.0', False): ('o-', '#b2182b', '1e4 PHIIoff'),
              ('10000.0', True): ('o--', '#ef8a62', '1e4 PHIIon'),
              ('100000.0', False): ('s-', '#2166ac', '1e5 PHIIoff'),
              ('100000.0', True): ('s--', '#67a9cf', '1e5 PHIIon')}
    for ax, nc in zip(axes, NCORE):
        ax.axhspan(R7 - R7E, R7 + R7E, color='#4daf4a', alpha=0.15)
        ax.axhline(R7, color='#4daf4a', lw=1); ax.axhline(R62, color='#984ea3', ls='--', lw=1)
        for (mass, phii), (st, col, lab) in styles.items():
            for fmix in [1.0, 4.0]:
                xs, ys, censx, censy = [], [], [], []
                for cf in CF_GRID:
                    hit = [r for r in ARMS if fnum(r['mCloud']) == float(mass)
                           and fnum(r['nCore']) == nc and fnum(r['coverFraction']) == cf
                           and fnum(r['cooling_boost_fmix']) == fmix
                           and (r['include_PHII'] == 'True') == phii]
                    if not hit:
                        continue
                    r = hit[0]; r2 = fnum(r['R2_at_best_7'])
                    if r['status_7'] in ('ok', 'ok_censored') and r2 is not None:
                        xs.append(cf); ys.append(r2)
                    else:  # age-censored: show at final R2, open marker (not a match)
                        censx.append(cf); censy.append(fnum(r['R2_final']))
                if xs:
                    ax.plot(xs, ys, st, color=col, lw=1.3, ms=5,
                            alpha=0.9 if fmix == 4.0 else 0.5,
                            label=f"{lab} fmix{fmix:g}" if nc == NCORE[0] else None)
                if censx:
                    ax.plot(censx, censy, 'x', color=col, ms=6, mew=1.5, alpha=0.6)
        ax.set_title(f"nCore={nc:g}", fontsize=10)
        ax.set_xlabel(r"$C_f$"); ax.set_xticks(CF_GRID); ax.set_ylim(0, 60)
        ax.invert_xaxis()
    axes[0].set_ylabel(r"$R_2$ at best-t in [1.5,2.5] Myr [pc]")
    axes[0].legend(fontsize=6, loc='upper right')
    fig.suptitle("A2 — value-at-age constraint: which $C_f$ reproduces the cavity "
                 "(green 7$\\pm$1, purple dashed 6.2 pc). ×  = age-censored (final $R_2$, not a match). "
                 "[interim stand-in for paper_Cf.plot_cf_constraint]", fontsize=9)
    watermark(fig); fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    out = os.path.join(HERE, 'A2_cf_constraint_bothbases.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- A3 match maps
def A3_match_maps():
    """chi2 heatmap over (nCore x C_f) per (mass x fmix x PHII) panel, per base.
    edge-min full-3 cells hatched; best (lowest chi2) arm per panel starred; NA dimmed."""
    for base in ['7', '62']:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
        panels = [(m, fx, ph) for m in ['10000.0', '100000.0'] for fx in [1.0, 4.0] for ph in [False, True]]
        for ax, (mass, fmix, phii) in zip(axes.flat, panels):
            grid = np.full((len(NCORE), len(CF_GRID)), np.nan)
            for r in ARMS:
                if fnum(r['mCloud']) != float(mass) or fnum(r['cooling_boost_fmix']) != fmix:
                    continue
                if (r['include_PHII'] == 'True') != phii:
                    continue
                if r[f'status_{base}'] not in ('ok', 'ok_censored'):
                    continue
                ii = NCORE.index(fnum(r['nCore'])); jj = CF_GRID.index(fnum(r['coverFraction']))
                grid[ii, jj] = fnum(r[f'chi2_min_{base}'])
            im = ax.imshow(np.log10(grid), origin='lower', aspect='auto', cmap='viridis_r',
                           vmin=-1, vmax=3)
            # star best (min chi2) cell
            if np.isfinite(grid).any():
                bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
                ax.plot(bj, bi, '*', color='white', ms=16, mec='k', mew=0.8)
            # hatch NA (age-censored) cells; annotate chi2
            for ii in range(len(NCORE)):
                for jj in range(len(CF_GRID)):
                    if not np.isfinite(grid[ii, jj]):
                        ax.add_patch(plt.Rectangle((jj - .5, ii - .5), 1, 1, fill=False,
                                                   hatch='xx', edgecolor='0.5', lw=0))
                        ax.text(jj, ii, 'cens', ha='center', va='center', fontsize=6, color='0.4')
                    else:
                        ax.text(jj, ii, f"{grid[ii,jj]:.0f}", ha='center', va='center',
                                fontsize=7, color='w' if grid[ii, jj] > 30 else 'k')
            ax.set_xticks(range(len(CF_GRID))); ax.set_xticklabels([f"{c:g}" for c in CF_GRID])
            ax.set_yticks(range(len(NCORE))); ax.set_yticklabels([f"{n:g}" for n in NCORE])
            mtag = '1e4/sfe.10' if mass == '10000.0' else '1e5/sfe.01'
            ax.set_title(f"{mtag}, fmix{fmix:g}, PHII{'on' if phii else 'off'}", fontsize=8)
            ax.set_xlabel(r"$C_f$"); ax.set_ylabel("nCore")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=r"$\log_{10}\chi^2$")
        base_lab = '7.0$\\pm$1.0 pc' if base == '7' else '6.2 pc'
        fig.suptitle(f"A3 — frozen-$\\chi^2$ match map over (nCore × $C_f$), cavity base {base_lab}. "
                     f"★ = best cell; hatched 'cens' = age-censored (no minimum). "
                     f"All full-3 cells edge-min at $C_f$=0.70 (see A4).", fontsize=10)
        watermark(fig)
        out = os.path.join(HERE, f'A3_match_map_base{base}.png')
        fig.savefig(out, dpi=125, bbox_inches='tight'); plt.close(fig)
        print("wrote", out)


# ---------------------------------------------------------------- A4 overshoot/gradient
def A4_overshoot():
    """R2_at_best - target vs nCore, one line per C_f, both bases (2 panels).
    matchable = filled; age-censored (extrapolation / no match) = open marker."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, base in zip(axes, ['7', '62']):
        target = R7 if base == '7' else R62
        for cf in CF_GRID:
            xs, ys, ox, oy = [], [], [], []
            for nc in NCORE:
                # average over mass/fmix/PHII at this (nCore, Cf); keep matchable vs censored separate
                vals = [fnum(r[f'R2_at_best_{base}']) for r in ARMS
                        if fnum(r['nCore']) == nc and fnum(r['coverFraction']) == cf
                        and r[f'status_{base}'] in ('ok', 'ok_censored')
                        and fnum(r[f'R2_at_best_{base}']) is not None]
                cens = [fnum(r['R2_final']) for r in ARMS
                        if fnum(r['nCore']) == nc and fnum(r['coverFraction']) == cf
                        and r[f'status_{base}'] not in ('ok', 'ok_censored')]
                if vals:
                    xs.append(nc); ys.append(np.mean(vals) - target)
                if cens:
                    ox.append(nc); oy.append(np.mean(cens) - target)
            if xs:
                ax.plot(xs, ys, 'o-', color=CF_COLORS[cf], lw=1.6, ms=7, label=f"$C_f$={cf:g}")
            if ox:  # open markers = age-censored (final R2, NOT a matched result)
                ax.plot(ox, oy, 'o', mfc='none', mec=CF_COLORS[cf], ms=8, mew=1.5)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xscale('log'); ax.set_xticks(NCORE); ax.set_xticklabels([f"{n:g}" for n in NCORE])
        ax.set_xlabel("nCore [cm$^{-3}$]")
        ax.set_title(f"cavity base {'7.0' if base=='7' else '6.2'} pc", fontsize=10)
    axes[0].set_ylabel(r"$R_2$(at best-t) $-$ target  [pc]  (>0 overshoot)")
    axes[0].legend(fontsize=8)
    fig.suptitle("A4 — cavity overshoot vs nCore, per $C_f$ (mean over mass/fmix/PHII). "
                 "Open markers = age-censored arms' final $R_2$ (extrapolation, NOT a match). "
                 "Gradient: overshoot falls with lower $C_f$ and higher nCore.", fontsize=9.5)
    watermark(fig); fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = os.path.join(HERE, 'A4_overshoot_gradient.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- A5 ranking
def A5_ranking(topn=12):
    """best-match ranking by frozen chi2 on each base (CSV + rendered table).
    pilot rows appended, marked different corner + matcher vintage."""
    rank_csv = os.path.join(HERE, 'A5_best_match_ranking.csv')
    with open(rank_csv, 'w', newline='') as fh:
        fh.write("# PROVISIONAL / IN-CONTAINER — INTERIM tier (PLAN §0.12). Frozen matcher, radii-only.\n")
        fh.write("# top matches by chi2 on each cavity base; pilot rows are a DIFFERENT corner + matcher vintage.\n")
        w = csv.writer(fh)
        w.writerow(['basis', 'rank', 'chi2', 'run_name', 'mCloud', 'sfe', 'nCore', 'coverFraction',
                    'fmix', 'PHII', 't_best', 'R2_at_best', 'rShell_at_best', 'source'])
        tables = {}
        for base in ['7', '62']:
            mm = [r for r in ARMS if r[f'status_{base}'] in ('ok', 'ok_censored')
                  and fnum(r[f'chi2_min_{base}']) is not None]
            mm.sort(key=lambda r: fnum(r[f'chi2_min_{base}']))
            rows = []
            for k, r in enumerate(mm[:topn], 1):
                w.writerow([base, k, round(fnum(r[f'chi2_min_{base}']), 2), r['run_name'],
                            r['mCloud'], r['sfe'], r['nCore'], r['coverFraction'],
                            r['cooling_boost_fmix'], r['include_PHII'], r.get(f't_best_{base}'),
                            r.get(f'R2_at_best_{base}'), r.get('rShell_at_best_7'), 'interim'])
                rows.append([k, f"{fnum(r[f'chi2_min_{base}']):.1f}",
                             f"{fnum(r['mCloud']):.0g}/{r['sfe']}", f"{fnum(r['nCore']):g}",
                             f"{fnum(r['coverFraction']):g}", f"{fnum(r['cooling_boost_fmix']):g}",
                             'on' if r['include_PHII'] == 'True' else 'off',
                             f"{fnum(r[f'R2_at_best_{base}']):.1f}", f"{fnum(r['rShell_at_best_7']):.1f}"])
            tables[base] = rows
        # pilot note row (different corner + matcher vintage — not comparable head-to-head)
        w.writerow(['pilot', '-', '~7.5', 'pilot_1e5_sfe001_n1e3_PL0_yesPHII_PISM1e4', '1e5', '0.01',
                    '1000', '0.89', 'n/a', 'on', '~2.0', '9.6', '20.9',
                    'pilot 2026-07-08 (DIFFERENT corner: PISM=1e4,nCore=1e3; fallback matcher)'])
    # rendered table PNG
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    cols = ['#', 'chi2', 'mass/sfe', 'nCore', 'Cf', 'fmix', 'PHII', 'R2@', 'rSh@']
    for ax, base in zip(axes, ['7', '62']):
        ax.axis('off')
        tbl = ax.table(cellText=tables[base], colLabels=cols, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.35)
        ax.set_title(f"A5 — best matches, cavity base {'7.0' if base=='7' else '6.2'} pc "
                     f"(frozen $\\chi^2$, radii-only)", fontsize=10)
    fig.text(0.5, 0.12, "Pilot (2026-07-08, Cf≈0.89, χ²≈7.5) is a DIFFERENT corner "
             "(PISM=1e4, nCore=1e3) and a different (fallback) matcher — not a head-to-head row.",
             ha='center', fontsize=8, style='italic')
    watermark(fig); fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(HERE, 'A5_best_match_ranking.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out, "and", rank_csv)


# ---------------------------------------------------------------- A6 coverage
def A6_coverage():
    """coverage of the full 8100-run grid: which axis values the interim (+pilot) sample."""
    axes_full = [
        ("mass×profile", ['1e5·PL0', '1e5·PL-1', '1e5·BE', '2e5·PL0', '2e5·PL-1', '2e5·BE',
                          '4e5·PL0', '4e5·PL-1', '4e5·BE', '5e5·PL0', '5e5·PL-1', '5e5·BE',
                          '1e6·PL0', '1e6·PL-1', '1e6·BE'],
         {'1e5·PL0'}, {'1e5·PL0'}),   # interim overlaps ONLY 1e5·PL0; its 1e4/sfe.10 pair is off-grid (see note)
        ("coverFraction", ['0.70', '0.77', '0.83', '0.85', '0.89', '0.95', '0.99', '1.0'],
         {'0.70', '0.85', '1.0'}, {'0.77', '0.83', '0.89', '0.95', '0.99'}),
        ("nCore", ['50', '100', '500', '1000', '10000'], {'50', '100', '500'}, {'1000'}),
        ("PISM", ['1e4', '1e5', '1e6'], {'1e5'}, {'1e4'}),
        ("nISM", ['0.1', '1'], {'1'}, {'1'}),
        ("cooling_fmix", ['1', '2.5', '4'], {'1', '4'}, set()),
        ("PHII", ['off', 'on'], {'off', 'on'}, {'on'}),
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for row, (name, vals, interim, pilot) in enumerate(axes_full):
        for k, v in enumerate(vals):
            face = '#cccccc'
            if v in interim:
                face = '#2166ac'
            ec = 'k'
            ax.add_patch(plt.Rectangle((k, row), 0.92, 0.82, facecolor=face,
                                       edgecolor=ec, lw=0.6))
            ax.text(k + 0.46, row + 0.41, v, ha='center', va='center', fontsize=7,
                    color='white' if v in interim else '0.25')
            if v in pilot:
                ax.plot(k + 0.46, row + 0.41, 'o', mfc='none', mec='#e41a1c', ms=16, mew=2)
        ax.text(-0.3, row + 0.41, name, ha='right', va='center', fontsize=9, weight='bold')
    ax.set_xlim(-3.5, 8.5); ax.set_ylim(-0.3, len(axes_full))
    ax.axis('off')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc='#2166ac', label='sampled by interim scan'),
                       Patch(fc='#cccccc', label='unsampled (waits on P21)'),
                       plt.Line2D([], [], marker='o', mfc='none', mec='#e41a1c', ms=12, mew=2,
                                  ls='', label='pilot corner (2026-07-08)')],
              loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8)
    fig.suptitle("A6 — coverage of the full ~8100-run grid. The interim scan samples ONE PISM, "
                 "ONE nISM, PL0 only, 3 of 8 $C_f$, 3 of 5 nCore.\n"
                 "Note: the interim '1e4/sfe.10' pair is NOT a full-grid mass row "
                 "(the grid holds M$_\\star$≈1000 via 1e5..1e6/decreasing-sfe); it is an "
                 "interim compact-cloud bracket. Most of the space is unsampled.", fontsize=9)
    watermark(fig); fig.tight_layout(rect=[0, 0.06, 1, 0.9])
    out = os.path.join(HERE, 'A6_coverage_map.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


if __name__ == '__main__':
    A1_trajectories()
    A2_constraint()
    A3_match_maps()
    A4_overshoot()
    A5_ranking()
    A6_coverage()
    print("\nATLAS complete ->", HERE)
