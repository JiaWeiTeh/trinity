#!/usr/bin/env python3
"""Schematic: why the old odeint bubble solve crashed ~1-in-3 runs by consuming
uninitialised memory, and how the solve_ivp rewrite makes it deterministic.

Pure schematic (no data). Run: python make_odeint_garbage_figure.py
Output: ../figs/odeint_uninitialised_memory.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figs', 'odeint_uninitialised_memory.png')

GREEN = '#2e7d32'
RED = '#c62828'
GRAY = '#9e9e9e'
BLUE = '#1565c0'
INK = '#212121'

fig = plt.figure(figsize=(13, 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def box(x, y, w, h, text, fc, ec, fs=10.5, tc=INK, weight='normal', round=True):
    style = "round,pad=0.3,rounding_size=1.4" if round else "square,pad=0.3"
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.8,
                       mutation_scale=1, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, weight=weight, zorder=3, wrap=True)


def arrow(x1, y1, x2, y2, color=INK, lw=2.0, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 mutation_scale=20, lw=lw, color=color, ls=ls, zorder=1))


ax.text(50, 97, 'The old odeint bubble solve: "consuming uninitialised memory"',
        ha='center', fontsize=16, weight='bold', color=INK)

# ============================================================================
# PANEL 1 — what the (N x 3) output array actually contains
# ============================================================================
ax.text(50, 91.5, 'Part 1 — what odeint actually hands back  (the T column of the (N×3) output array)',
        ha='center', fontsize=12, weight='bold', color=BLUE)

ncell = 14
kfail = 8
x0, y0, cw, ch = 8, 80, 6, 5
real_vals = ['3.0e4', '4.1e4', '5.6e4', '7.2e4', '9.0e4', '1.3e5', '2.0e5', '3.4e5']
junk_vals = ['0.0', '1.6e-318', '0.0', '?', '6e-321', '0.0']
for i in range(ncell):
    xc = x0 + i * cw
    if i < kfail:
        fc, ec, tc = '#e8f5e9', GREEN, GREEN
        val = real_vals[i]
    else:
        fc, ec, tc = '#ffebee', RED, RED
        val = junk_vals[i - kfail]
    ax.add_patch(Rectangle((xc, y0), cw - 0.5, ch, fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(xc + (cw - 0.5) / 2, y0 + ch / 2, val, ha='center', va='center',
            fontsize=7.5, color=tc, zorder=3)

# faint full-width "RAM block" band underneath
ax.add_patch(Rectangle((x0, y0 - 2.4), ncell * cw - 0.5, 1.8, fc='#f0f0f0',
             ec=GRAY, lw=1.0, ls=(0, (3, 2)), zorder=1))
ax.text(x0 + (ncell * cw) / 2, y0 - 1.5,
        'one raw RAM block handed to odeint — still holds leftover bytes from whatever used this memory before (NOT zeroed)',
        ha='center', va='center', fontsize=8.5, color=GRAY, style='italic')

# integrate arrow over the green region
ax.annotate('', xy=(x0 + kfail * cw - 1, y0 + ch + 2.2), xytext=(x0 + 1, y0 + ch + 2.2),
            arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2.2))
ax.text(x0 + (kfail * cw) / 2, y0 + ch + 3.4,
        'LSODA integrates inward, writing REAL T', ha='center', fontsize=9, color=GREEN)

# failure marker
xf = x0 + kfail * cw - 0.25
ax.plot([xf, xf], [y0 - 0.5, y0 + ch + 1.5], color=RED, lw=2.2, ls=(0, (2, 1.5)))
ax.text(xf + 0.5, y0 + ch + 3.4, 'istate ≠ 2:\nintegrator gives up',
        ha='left', va='center', fontsize=8.5, color=RED, weight='bold')

# garbage label
ax.text(x0 + kfail * cw + (ncell - kfail) * cw / 2, y0 + ch + 3.0,
        'never written → stale "garbage"', ha='center', fontsize=9, color=RED)

ax.text(50, 73.0,
        '"uninitialised memory" = these leftover bytes.  odeint returns the FULL array + a quiet status flag (no exception).',
        ha='center', fontsize=10, color=INK,
        bbox=dict(boxstyle='round,pad=0.4', fc='#fffde7', ec='#f9a825', lw=1.2))

# ============================================================================
# PANEL 2 — two flows side by side
# ============================================================================
ax.plot([50, 50], [3, 69], color='#cccccc', lw=1.2, ls=(0, (4, 3)))

# ---- LEFT lane: OLD (odeint) ----
lx = 25
ax.text(lx, 66.5, 'OLD  —  odeint: consume the whole array', ha='center',
        fontsize=12, weight='bold', color=RED)
box(lx - 21, 58.5, 42, 5.5,
    'old code IGNORES the status flag and uses ALL N rows\n("consuming garbage")', '#ffebee', RED, fs=10, tc=RED, weight='bold')
arrow(lx, 58.5, lx, 55.0, color=RED)
box(lx - 21, 49.0, 20, 5.5, 'n = Pb /(k_B·T)\n→ divide-by-zero on tail', '#fff', RED, fs=9, tc=INK)
box(lx + 1, 49.0, 20, 5.5, 'monotonic(T)\nscans the random tail', '#fff', RED, fs=9, tc=INK)
arrow(lx, 49.0, lx, 45.5, color=RED)
# diamond-ish decision
box(lx - 15, 39.0, 30, 5.8,
    'does the RANDOM tail happen to be monotonic?', '#f3e5f5', '#6a1b9a', fs=9.5, tc='#4a148c', weight='bold')
arrow(lx - 8, 39.0, lx - 12, 34.0, color=GREEN)
arrow(lx + 8, 39.0, lx + 12, 34.0, color=RED)
box(lx - 23, 28.5, 16, 5.0, 'yes (lucky):\nrun passes', '#e8f5e9', GREEN, fs=9, tc=GREEN)
box(lx + 6, 28.5, 17, 5.0, 'no: raise\nMonotonicError', '#ffebee', RED, fs=9, tc=RED, weight='bold')
arrow(lx + 14.5, 28.5, lx + 14.5, 24.0, color=RED)
box(lx - 1, 18.0, 24, 6.0,
    'run.py exit 1\n~100 lines & a module away', '#ffcdd2', RED, fs=9.5, tc=RED, weight='bold')
ax.text(lx + 11, 12.5, '≈ 1 in 3 identical runs', ha='center', fontsize=11,
        color=RED, weight='bold',
        bbox=dict(boxstyle='round,pad=0.35', fc='#fff', ec=RED, lw=1.6))
ax.text(lx, 8.0, 'same inputs, same numpy, single-threaded,\nfixed PYTHONHASHSEED — still flaked',
        ha='center', fontsize=8.5, color=GRAY, style='italic')

# ---- RIGHT lane: NEW (solve_ivp) ----
rx = 75
ax.text(rx, 66.5, 'NEW  —  solve_ivp: check, never consume', ha='center',
        fontsize=12, weight='bold', color=GREEN)
box(rx - 21, 58.5, 42, 5.5,
    'solve_ivp returns a success flag\n+ a continuous solution object', '#e8f5e9', GREEN, fs=10, tc=GREEN, weight='bold')
arrow(rx, 58.5, rx, 54.5, color=GREEN)
box(rx - 13, 48.5, 26, 5.5, 'sol.success ?', '#e3f2fd', BLUE, fs=11, tc=BLUE, weight='bold')
arrow(rx - 9, 48.5, rx - 14, 43.5, color=RED)
arrow(rx + 9, 48.5, rx + 14, 43.5, color=GREEN)
box(rx - 25, 37.0, 20, 6.5,
    'no → raise\nBubbleSolverError\n(deterministic, caught\n& penalised by fsolve)', '#fff3e0', '#e65100', fs=8.3, tc='#e65100', weight='bold')
box(rx + 4, 37.0, 21, 6.5,
    'yes → sample the SMOOTH\ncontinuous solution\n(never touches an\nun-integrated region)', '#e8f5e9', GREEN, fs=8.3, tc=GREEN, weight='bold')
arrow(rx - 15, 37.0, rx - 4, 26.5, color='#e65100')
arrow(rx + 14.5, 37.0, rx + 4, 26.5, color=GREEN)
box(rx - 17, 20.0, 34, 6.0,
    'deterministic outcome\nNO garbage is ever read', '#c8e6c9', GREEN, fs=10.5, tc=GREEN, weight='bold')
ax.text(rx, 12.8, 'byte-identical dictionary.jsonl every run', ha='center',
        fontsize=11, color=GREEN, weight='bold',
        bbox=dict(boxstyle='round,pad=0.35', fc='#fff', ec=GREEN, lw=1.6))
ax.text(rx, 8.0, 'verified: 124-snapshot run, 128,114 fields,\nworst diff 0.0 vs the pre-rewrite code',
        ha='center', fontsize=8.5, color=GRAY, style='italic')

fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')
print('wrote', os.path.normpath(OUT))
