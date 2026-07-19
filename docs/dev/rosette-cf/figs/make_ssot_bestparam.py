#!/usr/bin/env python3
"""P21 best-parameter visualisation from the SSOT (7830-run Helix survey, 2026-07-14).

fate='ongoing' ONLY — a recollapsing bubble is not a stable H II region, so it cannot
be "the Rosette" (see PLAN §0.3 adjudication). Recollapsed/dispersed cells are shown but
excluded from every minimum. Radii-only frozen χ² (R2 7±1, rShell 19±2; 7 pc base);
θ is NOT yet calibrated on L_X -> this ranks radii, not the full physics (P6 finalizes).

Reads the SSOT: paper/rosette/plots/{summary,match,trajectory_points}.csv (local mount).
Command: python docs/dev/rosette-cf/figs/make_ssot_bestparam.py
Outputs: docs/dev/rosette-cf/figs/ssot_{matchmap,cf_curve,ranking,bestcell_traj}.png
"""
import csv, os, re, statistics
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, '..', '..', '..', '..', 'paper', 'rosette', 'plots')
NCORE = [100.0, 1000.0, 10000.0]
CF = [0.77, 0.83, 0.89, 0.95, 0.99]
WM = ("SSOT — 7830-run Helix survey (2026-07-14). Radii-only frozen χ² (7 pc base); "
      "θ NOT yet calibrated on L_X → P6. fate='ongoing' only (recollapsing bubbles excluded).")


def f(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def load():
    s = {r['run_name']: r for r in csv.DictReader(open(os.path.join(PLOTS, 'summary.csv')))}
    m = {r['run_name']: r for r in csv.DictReader(open(os.path.join(PLOTS, 'match.csv')))}
    J = []
    for rn, mr in m.items():
        c = f(mr['chi2_min']); sr = s.get(rn, {})
        J.append(dict(rn=rn, chi2=(None if c is None or c == float('inf') else c),
                      R2=f(mr.get('R2_at_best')), nCore=f(mr['nCore']), cf=f(mr['coverFraction']),
                      mCloud=f(mr['mCloud']), profile=sr.get('profile'), phii=mr['phii'],
                      fmix=f(sr.get('fmix')), fate=sr.get('fate'), v2=f(mr.get('v2_kms_at_best')),
                      rSh=f(mr.get('rShell_at_best'))))
    return J


def watermark(fig):
    fig.text(0.5, 0.006, WM, ha='center', va='bottom', fontsize=7, color='#444', wrap=True)


def matchmap(J):
    """min ongoing χ² over (nCore × C_f) for the PHYSICAL pair (mCloud 1e5),
    faceted by profile × PHII. Recollapsed-only cells hatched. Best cell starred."""
    phys = [r for r in J if r['mCloud'] == 1e5]
    profs = ['PL0', 'PL-1', 'BE']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    gmin = None
    for row, phii in enumerate(['False', 'True']):
        for col, prof in enumerate(profs):
            ax = axes[row][col]
            grid = np.full((len(NCORE), len(CF)), np.nan)
            for i, nc in enumerate(NCORE):
                for j, cf in enumerate(CF):
                    ong = [r['chi2'] for r in phys if r['profile'] == prof and r['phii'] == phii
                           and r['nCore'] == nc and r['cf'] == cf
                           and r['fate'] == 'ongoing' and r['chi2'] is not None]
                    if ong:
                        grid[i, j] = min(ong)
            im = ax.imshow(np.log10(grid), origin='lower', aspect='auto', cmap='viridis_r',
                           vmin=-0.5, vmax=3)
            if np.isfinite(grid).any():
                bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
                ax.plot(bj, bi, '*', color='white', ms=17, mec='k', mew=0.9)
            for i in range(len(NCORE)):
                for j in range(len(CF)):
                    if not np.isfinite(grid[i, j]):
                        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                                   hatch='xxx', edgecolor='0.6', lw=0))
                        ax.text(j, i, 'no\nongoing', ha='center', va='center', fontsize=6, color='0.45')
                    else:
                        ax.text(j, i, f"{grid[i,j]:.0f}", ha='center', va='center',
                                fontsize=8, color='w' if grid[i, j] > 20 else 'k')
            ax.set_xticks(range(len(CF))); ax.set_xticklabels(CF, fontsize=8)
            ax.set_yticks(range(len(NCORE))); ax.set_yticklabels([f"{n:g}" for n in NCORE], fontsize=8)
            ax.set_title(f"{prof}, PHII {'on' if phii=='True' else 'off'}", fontsize=9)
            if row == 1: ax.set_xlabel(r"$C_f$")
            if col == 0: ax.set_ylabel(f"PHII {'on' if phii=='True' else 'off'}\n"+r"nCore [cm$^{-3}$]", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=r"$\log_{10}\chi^2$ (ongoing min)")
    fig.suptitle("SSOT best-parameter map — PHYSICAL Rosette pair (M$_{\\rm cloud}$=10$^5$, sfe=0.01), "
                 "min ongoing χ² over (nCore × $C_f$).\n★ = best cell. Hatched = no ongoing run "
                 "(all recollapse/disperse). Best overall: BE, PHII off, nCore 10$^3$, $C_f$≈0.89.",
                 fontsize=10)
    watermark(fig)
    out = os.path.join(HERE, 'ssot_matchmap.png'); fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig); print("wrote", out)


def cf_curve(J):
    """min ongoing χ² vs C_f — the interior optimum, physical pair vs whole grid."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pool, lab, col in [([r for r in J if r['mCloud'] == 1e5], 'physical pair (1e5)', '#b2182b'),
                           (J, 'whole grid (any mass)', '#2166ac')]:
        ys = []
        for cf in CF:
            ong = [r['chi2'] for r in pool if r['cf'] == cf and r['fate'] == 'ongoing' and r['chi2'] is not None]
            ys.append(min(ong) if ong else np.nan)
        ax.plot(CF, ys, 'o-', color=col, lw=1.8, ms=8, label=lab)
    ax.axvline(0.89, color='0.6', ls=':', lw=1.2)
    ax.text(0.89, ax.get_ylim()[1]*0.9, ' interior optimum\n C_f≈0.89', fontsize=8, color='0.3')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel(r"covering fraction $C_f$"); ax.set_ylabel(r"min ongoing $\chi^2$ (radii-only)")
    ax.set_title("SSOT — $C_f$ is an INTERIOR optimum (~0.89), not the interim edge-min.\n"
                 "Too much leak ($C_f{<}0.89$) collapses early; too little overshoots then recollapses.",
                 fontsize=9.5)
    ax.legend(); ax.set_xticks(CF)
    watermark(fig); fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(HERE, 'ssot_cf_curve.png'); fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def ranking(J):
    ong = sorted([r for r in J if r['fate'] == 'ongoing' and r['chi2'] is not None], key=lambda r: r['chi2'])[:15]
    rk = os.path.join(HERE, 'ssot_best_match_ranking.csv')
    with open(rk, 'w', newline='') as fh:
        fh.write("# SSOT (7830-run, 2026-07-14) top ongoing matches by radii-only frozen chi2 (7pc). theta uncalibrated (P6).\n")
        w = csv.writer(fh); w.writerow(['rank', 'chi2', 'mCloud', 'nCore', 'coverFraction', 'profile', 'phii', 'fmix', 'R2', 'rShell', 'v2_kms', 'run_name'])
        for k, r in enumerate(ong, 1):
            w.writerow([k, round(r['chi2'], 2), f"{r['mCloud']:.0g}", f"{r['nCore']:.0g}", r['cf'], r['profile'],
                        'on' if r['phii'] == 'True' else 'off', f"{r['fmix']:g}", round(r['R2'], 2),
                        round(r['rSh'], 1), round(r['v2'], 1), r['rn']])
    cols = ['#', 'χ²', 'mass', 'nCore', 'C_f', 'prof', 'PHII', 'fmix', 'R2', 'rSh', 'v2']
    cells = [[k, f"{r['chi2']:.1f}", f"{r['mCloud']:.0g}", f"{r['nCore']:.0g}", f"{r['cf']:g}", r['profile'],
              'on' if r['phii'] == 'True' else 'off', f"{r['fmix']:g}", f"{r['R2']:.1f}", f"{r['rSh']:.1f}", f"{r['v2']:.1f}"]
             for k, r in enumerate(ong, 1)]
    fig, ax = plt.subplots(figsize=(11, 5.5)); ax.axis('off')
    t = ax.table(cellText=cells, colLabels=cols, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.4)
    ax.set_title("SSOT — top-15 ONGOING (expanding) matches, radii-only frozen χ² (7 pc). "
                 "The physical pair (1e5) tops it at $C_f$≈0.89; note the slow v2~1 km/s.", fontsize=9.5)
    watermark(fig); fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(HERE, 'ssot_ranking.png'); fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out, "and", rk)


def bestcell_traj(J):
    """paper_Cf-style R2/v2/rShell(t) for the best cell's C_f sweep, from trajectory_points.csv."""
    best = min([r for r in J if r['fate'] == 'ongoing' and r['chi2'] is not None], key=lambda r: r['chi2'])
    cellpat = re.compile(re.escape(re.sub(r'coverFraction[0-9p]+', 'coverFraction<CF>', best['rn']))
                         .replace('coverFraction<CF>', r'coverFraction[0-9p]+'))
    s = {r['run_name']: r for r in csv.DictReader(open(os.path.join(PLOTS, 'summary.csv')))}
    members = sorted([rn for rn in s if cellpat.fullmatch(rn)], key=lambda rn: f(s[rn]['coverFraction']))
    traj = defaultdict(lambda: dict(t=[], R2=[], v2=[], rSh=[]))
    want = set(members)
    with open(os.path.join(PLOTS, 'trajectory_points.csv')) as fh:
        for r in csv.DictReader(fh):
            rn = r['run_name']
            if rn in want:
                traj[rn]['t'].append(f(r['t'])); traj[rn]['R2'].append(f(r['R2']))
                traj[rn]['v2'].append(f(r['v2'])); traj[rn]['rSh'].append(f(r['rShell']))
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    PC = 1.0 / 1.0227
    for rn in members:
        cf = f(s[rn]['coverFraction']); fate = s[rn]['fate']
        t = np.array(traj[rn]['t']); order = np.argsort(t); t = t[order]
        R2 = np.array(traj[rn]['R2'])[order]; v2 = np.array(traj[rn]['v2'])[order] * PC
        rSh = np.array(traj[rn]['rSh'])[order]
        col = cmap((cf - 0.77) / (0.99 - 0.77))
        style = dict(color=col, lw=2.2 if fate == 'ongoing' else 1.3,
                     ls='-' if fate == 'ongoing' else ':', alpha=1.0 if fate == 'ongoing' else 0.5)
        lab = f"$C_f$={cf:g}" + (" ✓ongoing" if fate == 'ongoing' else f" ({fate})")
        axes[0].plot(t, R2, label=lab, **style)
        axes[1].plot(t, v2, **style)
        axes[2].plot(t, rSh, **style)
    # obs bands
    axes[0].axhspan(4.5, 7.0, color='green', alpha=0.10); axes[0].axhline(7, color='#4daf4a', lw=1)
    axes[0].axhline(6.2, color='#984ea3', ls='--', lw=1); axes[0].set_ylim(0, 25); axes[0].set_ylabel('R2 (cavity) [pc]')
    axes[0].set_title('cavity R2(t) — 4.5–7 pc band')
    axes[1].axhspan(4.5-2, 4.5+2, color='#0072B2', alpha=0.10); axes[1].axhline(4.5, color='#0072B2', lw=1)
    axes[1].set_ylim(0, 20); axes[1].set_ylabel('v2 [km/s]'); axes[1].set_title('v2(t) — neutral-shell 4.5 km/s band')
    axes[2].axhspan(19-2, 19+2, color='blue', alpha=0.10); axes[2].axhline(19, color='blue', lw=1)
    axes[2].set_ylim(0, 40); axes[2].set_ylabel('rShell [pc]'); axes[2].set_title('rShell(t) — HII 19 pc band')
    for ax in axes:
        ax.axvspan(1.5, 2.5, color='0.85', alpha=0.5); ax.set_xlabel('t [Myr]'); ax.set_xlim(0, 3)
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle("SSOT best cell — physical pair (1e5), nCore 10$^3$, BE, PHII off, fmix 4: the $C_f$ sweep.\n"
                 "Only $C_f$=0.89 stays EXPANDING to the age window (solid); others recollapse (dotted). "
                 "[from trajectory_points.csv; the real paper_Cf on the pulled dicts adds house style]", fontsize=9.5)
    watermark(fig); fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    out = os.path.join(HERE, 'ssot_bestcell_traj.png'); fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out, "| best cell members:", len(members))
    print("  PULL these dicts for the real paper_Cf/paper_Rosette:")
    for rn in members:
        print("   ", rn)


if __name__ == '__main__':
    J = load()
    matchmap(J); cf_curve(J); ranking(J); bestcell_traj(J)
