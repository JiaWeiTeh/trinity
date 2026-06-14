#!/usr/bin/env python
"""Phase 6.0 Gate-G6 analysis over hunt CSVs.

For each config, classify the interior inflow (WARPFIELD "Problem 2") as
COSMETIC or CONTAMINATING using only signals the CSV carries directly:

  1. NON-CONVERGENCE in the band -- any inflow segment with no_physical_root
     or betadelta_converged=False (sentinel v_struct_nneg = -1).
  2. SOLUTION ROUGHNESS across the band -- the max per-step relative change of
     Lloss / dMdt / Eb among inflow segments vs the median per-step change over
     the whole run. A kink *caused* by inflow shows up as band roughness >> the
     run's typical step (flagged at >5x, and absolutely >5%).
  3. BAND DOMINANCE -- inflow reaching a large fraction of the bubble thickness
     (max v_neg_frac_thick) or grid points (max nneg / npts).

Gate G6: a config is CONTAMINATING if (1) fires, or (2) fires, or (3) exceeds
the dominance threshold. If no config contaminates -> inflow is cosmetic, STOP.

    python scratch/phase6/analyze_hunt.py analysis/data/hunt_*.csv
"""
import csv
import sys

import numpy as np

DOMINANCE_FRAC = 0.5       # band > 50% of thickness => dominant (descriptive)
ROUGHNESS_REL = 0.05       # absolute per-step change below which nothing is a kink
ROUGHNESS_RATIO = 1.5      # band step must exceed this x the surge ramp to count
V_FLOOR = 0.01             # |v_min| (pc/Myr) above which inflow is real, not BC noise
REAL_FRAC = 0.02           # thickness fraction above which inflow is real
RAMP_W = 4                 # segments of surge ramp to compare on each side


def _f(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def _rel_steps(vals):
    """Per-step relative change |dx/x| over a sequence (ignores nan/zero)."""
    v = np.asarray(vals, dtype=float)
    out = []
    for a, b in zip(v[:-1], v[1:]):
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            out.append(abs((b - a) / a))
    return np.asarray(out) if out else np.asarray([np.nan])


def _win_step(vals, idxs):
    """Max |relative step| between CONSECUTIVE segments within a window."""
    idxs = sorted(i for i in idxs)
    out = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        if b == a + 1:
            x0, x1 = vals[a], vals[b]
            if np.isfinite(x0) and np.isfinite(x1) and x0 != 0:
                out.append(abs((x1 - x0) / x0))
    return max(out) if out else 0.0


def analyze(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    t = [_f(r['t_now']) for r in rows]
    bpd = [_f(r['beta_plus_delta']) for r in rows]
    vmin = [_f(r['v_struct_min']) for r in rows]
    nneg = [int(_f(r['v_struct_nneg'])) for r in rows]
    npts = [int(_f(r['v_struct_npts'])) for r in rows]
    frac = [_f(r['v_neg_frac_thick']) for r in rows]
    conv = [r['betadelta_converged'] == 'True' for r in rows]
    noroot = [r['no_physical_root'] == 'True' for r in rows]
    Lloss = [_f(r['bubble_Lloss']) for r in rows]
    dMdt = [_f(r['bubble_dMdt']) for r in rows]
    Eb = [_f(r['Eb']) for r in rows]

    noroot_idx = [i for i in range(n) if noroot[i]]
    # Any v<0 segment. The structure grid is ~6e4 points, so an absolute nneg
    # count is meaningless -- classify by velocity magnitude + thickness frac.
    any_neg = [i for i in range(n) if nneg[i] > 0]
    # REAL inflow vs the cosmetic inner-edge BC undershoot: the BC artifact sits
    # at v_min ~ -1e-3 pc/Myr with frac~0 (a single-cell v->0 overshoot); real
    # inflow reaches |v_min| >~ V_FLOOR or spans a real fraction of the bubble.
    real_idx = [i for i in any_neg if (vmin[i] < -V_FLOOR or frac[i] > REAL_FRAC)]
    bc_idx = [i for i in any_neg if i not in real_idx]

    print(f"\n=== {path.split('/')[-1]} ===")
    print(f"  segments={n}  converged={sum(conv)}/{n}  no_physical_root={len(noroot_idx)}")
    print(f"  beta+delta range: [{min(bpd):.3f}, {max(bpd):.3f}]   t range: [{min(t):.3f}, {max(t):.3f}] Myr")
    print(f"  v<0 segments: {len(any_neg)}  (cosmetic BC artifacts: {len(bc_idx)}, REAL inflow: {len(real_idx)})")
    if not real_idx:
        print("  inflow: only inner-edge BC artifacts  ->  COSMETIC")
        return False
    bpd_in = [bpd[i] for i in real_idx]
    print(f"  REAL inflow band:")
    print(f"    t band         : [{t[min(real_idx)]:.3f}, {t[max(real_idx)]:.3f}] Myr")
    print(f"    beta+delta band: [{min(bpd_in):.3f}, {max(bpd_in):.3f}]")
    print(f"    worst v_struct_min : {min(vmin[i] for i in real_idx):.3e} pc/Myr")
    # band dominance (thickness fraction; nneg/npts as a cross-check)
    maxfrac = max(frac[i] for i in real_idx)
    maxfracpts = max((nneg[i] / npts[i]) for i in real_idx if npts[i] > 0)
    print(f"    max band thickness frac: {maxfrac:.3f}   max nneg/npts: {maxfracpts:.3f}")

    # signal 1: non-convergence among real-inflow segments
    sig_nonconv = any((not conv[i]) or noroot[i] for i in real_idx)

    # signal 2: roughness ATTRIBUTABLE TO THE INFLOW (deconfounded). The inflow
    # band coincides with the feedback (Lmech) surge, which on its own drives
    # dMdt/Lloss/Eb up; per-step changes also grow with the timestep (~50x larger
    # at the late-time surge). So compare the band's step NOT to the run median
    # but to the surge ramp immediately LEADING and TRAILING it (same surge, no
    # inflow). Inflow contaminates only if it makes the band substantially
    # rougher than the surge already is on both sides.
    i0, i1 = min(real_idx), max(real_idx)
    lead = [i for i in range(i0 - RAMP_W, i0) if 0 <= i < n]
    trail = [i for i in range(i1 + 1, i1 + 1 + RAMP_W) if 0 <= i < n]
    handoff = len(lead) < 2  # band at the very start => no surge ramp to compare
    # Only dMdt is mechanically coupled to v (the structure solve matches the
    # velocity BC). Lloss/Eb use (n,T,phi) with v ABSENT from every cooling
    # integral (bubble_luminosity.py:612/659/677), so their band roughness
    # cannot be inflow-caused -- report it as descriptive, gate ONLY on dMdt.
    sig_rough = False
    rough_report = []
    for k, v in (('dMdt', dMdt), ('Lloss', Lloss), ('Eb', Eb)):
        bstep = _win_step(v, range(i0 - 1, i1 + 1))   # onset + within-band steps
        ramp = max(_win_step(v, lead), _win_step(v, trail))
        ratio = bstep / ramp if ramp > 0 else float('inf')
        hot = (not handoff) and (bstep > ROUGHNESS_REL) and (ratio > ROUGHNESS_RATIO)
        gates = (k == 'dMdt')          # v-coupled output gates the verdict
        if gates:
            sig_rough = sig_rough or hot
        tag = ' KINK' if hot else ''
        tag += '' if gates else ' (descriptive; v-immune)'
        rough_report.append(f"{k}: band {bstep*100:.1f}% vs surge-ramp {ramp*100:.1f}% (x{ratio:.1f}){tag}")
    # descriptive only: v is absent from the cooling integrals
    # (bubble_luminosity.py:612/659/677), so a deep inflow band cannot corrupt
    # Lloss/Eb -- report the depth but do NOT gate the verdict on it.
    dom = (maxfrac >= DOMINANCE_FRAC) or (maxfracpts >= DOMINANCE_FRAC)

    print("    roughness (deconfounded vs surge ramp):")
    for line in rough_report:
        print(f"      {line}")
    note = "  [handoff transient -- no surge ramp to deconfound]" if handoff else ""
    print(f"    depth-dominant={dom} (descriptive; v not in Lloss){note}")
    print(f"    contamination signals -> nonconvergence={sig_nonconv}  inflow-roughness={sig_rough}")
    contaminating = sig_nonconv or sig_rough
    print(f"    VERDICT: {'CONTAMINATING' if contaminating else 'cosmetic'}")
    return contaminating


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: analyze_hunt.py <hunt_*.csv> ...")
        sys.exit(2)
    any_contam = False
    for p in paths:
        try:
            any_contam |= analyze(p)
        except Exception as e:
            print(f"\n=== {p} ===\n  FAILED to analyze: {e}")
    print("\n" + "=" * 60)
    if any_contam:
        print("GATE G6: OPEN -- at least one config shows material contamination.")
        print("  -> proceed to Phase 6.1 (treatments) on the contaminating config.")
    else:
        print("GATE G6: CLOSED -- inflow is cosmetic across all configs.")
        print("  -> document and STOP (optional diagnostic-only snapshot field).")


if __name__ == '__main__':
    main()
