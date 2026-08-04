#!/usr/bin/env python3
"""Figures for the phase-1a initialisation investigation (M43 probe).

Reads ONLY the committed CSVs in docs/dev/phase1a-init/data/ and writes
PNGs to docs/dev/phase1a-init/figures/. Run from the repo root:

    python docs/dev/phase1a-init/harness/make_figures.py

Analytic overlays (documented in FINDINGS.md §Q6):
  - Weaver adiabatic:  R(t) = 0.76 (Lw t^3 / rho)^(1/5)   [AU units]
  - Spitzer D-type:    R(t) = Rst (1 + 7 c_i t / (4 Rst))^(4/7)
Constants below are for the M43 probe config (mCloud=300, sfe=0.01,
nCore=8.7e3 cm^-3): Lw and rho from the SB99 default table at t=0 and
nCore*mu_convert; Rst from Q=1.6185e47 s^-1, alphaB=2.59e-13 cm^3/s.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = False

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, '..', 'data')
FIGS = os.path.join(HERE, '..', 'figures')

KMS = 0.977792          # km/s per pc/Myr
LW_AU = 6.0845e4        # M43 probe wind Lmech at t=0 [Msun pc^2/Myr^3]
RHO_AU = 301.008        # 8.7e3 cm^-3 * mu_convert -> Msun/pc^3
RST_PC = 0.0406         # Stromgren radius for Q=1.6185e47, n=8.7e3
CI_KMS = 10.0           # ionized-gas sound speed for the Spitzer overlay


def load(name):
    path = os.path.join(DATA, name)
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    r = csv.DictReader(lines)
    cols = {k: [] for k in r.fieldnames}
    for row in r:
        for k, v in row.items():
            cols[k].append(float(v) if k != 'current_phase' else v)
    return {k: (np.array(v) if k != 'current_phase' else v) for k, v in cols.items()}


def weaver_R(t):
    return 0.76 * (LW_AU * t**3 / RHO_AU) ** 0.2


def spitzer_R(t):
    ci = CI_KMS / KMS
    return RST_PC * (1 + 7 * ci * t / (4 * RST_PC)) ** (4.0 / 7.0)


def fig_convergence():
    runs = [
        ('m43_probe.csv', 'baseline (SEG=3e-5, hack on)', 'C0'),
        ('m43_seg1e-5.csv', 'SEG=1e-5, hack on', 'C1'),
        ('m43_seg3e-6.csv', 'SEG=3e-6, hack on', 'C2'),
        ('m43_noapprox.csv', 'SEG=3e-5, hack ablated', 'C3'),
        ('m43_tol1e-8.csv', 'rtol 1e-8/atol 1e-11', 'C4'),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, label, c in runs:
        if not os.path.exists(os.path.join(DATA, name)):
            continue
        d = load(name)
        ax1.loglog(d['t_now'] * 1e6, d['R2'], color=c, label=label, lw=1.4)
        ax2.loglog(d['t_now'] * 1e6, d['v2_kms'], color=c, lw=1.4)
    t = np.logspace(-2, 5, 300) * 1e-6
    ax1.loglog(t * 1e6, weaver_R(t), 'k--', lw=1, label='Weaver adiabatic')
    ax1.loglog(t * 1e6, spitzer_R(t), 'k:', lw=1.2, label='Spitzer D-type')
    ax2.loglog(t[1:] * 1e6, np.gradient(weaver_R(t), t)[1:] * KMS, 'k--', lw=1)
    ax2.loglog(t[1:] * 1e6, np.gradient(spitzer_R(t), t)[1:] * KMS, 'k:', lw=1.2)
    ax1.errorbar([2.1e4], [0.153], yerr=[[0.011], [0.011]],
                 xerr=[[0.4e4], [1.3e4]], fmt='r*', ms=12, label='M43 observed')
    ax2.errorbar([2.1e4], [5.0], yerr=[[0.5], [1.6]], xerr=[[0.4e4], [1.3e4]],
                 fmt='r*', ms=12)
    ax1.set_xlabel('t [yr]'); ax1.set_ylabel('R2 [pc]')
    ax2.set_xlabel('t [yr]'); ax2.set_ylabel('v2 [km/s]')
    ax1.set_ylim(1e-5, 2); ax2.set_ylim(1, 5e3)
    ax1.legend(fontsize=7, loc='upper left')
    fig.suptitle('M43 probe: trajectory vs purely numerical knobs (nothing physical varied)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'convergence.png'), dpi=135)


def fig_budget():
    d = load('m43_probe.csv')
    t, R = d['t_now'], d['R2']
    p = d['p_shell']
    pdw = d['F_ram_wind']
    FP = 4 * np.pi * R**2 * d['P_drive']
    Fg = d['F_grav']
    def cum(F):
        return np.concatenate([[0], np.cumsum(np.diff(t) * (F[1:] + F[:-1]) / 2)])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(t * 1e6, p, 'C0', lw=1.6, label='shell momentum  m v')
    ax.loglog(t * 1e6, cum(pdw), 'C1', lw=1.3, label=r'wind impulse  $\int \dot p_w dt$')
    ax.loglog(t * 1e6, cum(FP), 'C2', lw=1.3, label=r'pressure impulse  $\int 4\pi R^2 P_{drive} dt$')
    ax.loglog(t * 1e6, cum(Fg), 'C3', lw=1.0, label=r'gravity impulse  $\int F_{grav} dt$')
    ax.axvline(30, color='k', ls=':', lw=0.8)
    ax.annotate('end of segment 1', (33, 1e-5), fontsize=8, rotation=90)
    ax.set_xlabel('t [yr]'); ax.set_ylabel('momentum [Msun pc/Myr]')
    ax.set_title('M43 probe baseline: momentum budget\n(shell exits segment 1 with 2.9e5x the wind impulse)')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'momentum_budget.png'), dpi=135)


def fig_scales():
    runs = [('mass_3e3.csv', 'mCloud=3e3'), ('mass_3e4.csv', 'mCloud=3e4'),
            ('mass_3e5.csv', 'mCloud=3e5'), ('mass_3e6.csv', 'mCloud=3e6'),
            ('m43_probe.csv', 'mCloud=3e2 (probe)')]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, label in runs:
        if not os.path.exists(os.path.join(DATA, name)):
            continue
        d = load(name)
        sel = d['t_now'] <= 4e-3
        ax1.loglog(d['t_now'][sel] * 1e6, d['R2'][sel], label=label, lw=1.3)
        ax2.loglog(d['t_now'][sel] * 1e6, d['v2_kms'][sel], lw=1.3)
    ax1.set_xlabel('t [yr]'); ax1.set_ylabel('R2 [pc]')
    ax2.set_xlabel('t [yr]'); ax2.set_ylabel('v2 [km/s]')
    ax1.legend(fontsize=8)
    fig.suptitle('Early trajectory vs mCloud (sfe=0.01, nCore=8.7e3): '
                 'segment-1 exit state is mass-independent')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'mass_sweep.png'), dpi=135)


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)
    fig_convergence()
    fig_budget()
    fig_scales()
    print('figures written to', FIGS)
