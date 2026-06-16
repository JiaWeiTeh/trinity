#!/usr/bin/env python
"""Phase 6.0 contamination hunt -- per-segment bubble-velocity diagnostics.

Non-invasive: wraps ``solve_betadelta_pure`` for the duration of a full TRINITY
run and records, per accepted energy-phase segment, the bubble-structure
velocity health (``v_struct_min``, ``v_struct_nneg`` count, inflow thickness
fraction), convergence, Lloss/dMdt/Eb, and beta+delta. The CSV schema is a
superset of ``docs/dev/data/stalling_*.csv`` (adds ``v_struct_npts``,
``v_neg_frac_thick``, ``no_physical_root``) so the old runs stay comparable.

Goal (Phase 6.0, docs/dev/archive/betadelta/HYBR_PLAN.md): hunt a regime where interior
inflow (WARPFIELD "Problem 2", driven by (beta+delta)/t in dvdr at
bubble_luminosity.py:1150) stops being cosmetic -- i.e. non-convergence in the
band, a kink in Lloss/dMdt/Eb across it, or the band growing to dominate the
bubble thickness. Touches no production code.

    python docs/dev/archive/betadelta/velstruct/hunt.py <param> --out docs/dev/data/hunt_<name>.csv
    python docs/dev/archive/betadelta/velstruct/hunt.py <param> --validate-only
"""
import argparse
import csv
import dataclasses
import sys

import numpy as np

FIELDS = [
    't_now', 'cool_beta', 'cool_delta', 'beta_plus_delta',
    'Pb', 'bubble_dMdt', 'Lmech_total', 'Lmech_W', 'Lmech_SN',
    'bubble_Lgain', 'bubble_Lloss', 'cooling_ratio',
    'v_struct_min', 'v_struct_nneg', 'v_struct_npts', 'v_neg_frac_thick',
    'R2', 'v2', 'Eb', 'bubble_Tavg', 'c_sound',
    'no_physical_root', 'betadelta_converged',
]


def _pv(params, key):
    try:
        return float(params[key].value)
    except Exception:
        return float('nan')


def _frac(res):
    """Inflow thickness-fraction of a result's structure (None if no structure)."""
    bp = res.bubble_properties
    if bp is None:
        return None
    v = np.asarray(getattr(bp, 'bubble_v_arr', []), dtype=float)
    r = np.asarray(getattr(bp, 'bubble_r_arr', []), dtype=float)
    if v.size < 2 or r.size != v.size:
        return 0.0
    neg = v < 0
    rspan = abs(float(r[-1]) - float(r[0]))
    if not neg.any() or rspan <= 0:
        return 0.0
    rn = r[neg]
    return abs(float(rn.max()) - float(rn.min())) / rspan


def _row(params, res):
    bp = res.bubble_properties
    v_arr = getattr(bp, 'bubble_v_arr', None) if bp is not None else None
    if v_arr is not None:
        v = np.asarray(v_arr, dtype=float)
        r = np.asarray(bp.bubble_r_arr, dtype=float)
        npts = int(v.size)
        vmin = float(np.nanmin(v)) if npts else float('nan')
        neg = v < 0
        nneg = int(np.count_nonzero(neg))
        # thickness fraction spanned by the inflow band (r-extent of v<0 / total)
        if nneg and r.size == npts and npts > 1:
            rspan = abs(float(r[-1]) - float(r[0]))
            rneg = r[neg]
            frac = abs(float(rneg.max()) - float(rneg.min())) / rspan if rspan > 0 else float('nan')
        else:
            frac = 0.0
        dMdt = float(bp.bubble_dMdt)
        Tavg = float(bp.bubble_Tavg)
    else:
        # no_physical_root / failed solve: props is None. Record the gap as a
        # sentinel (nneg=-1) -- non-convergence in the band is itself a Gate-G6
        # contamination signal.
        npts, vmin, nneg, frac = 0, float('nan'), -1, float('nan')
        dMdt, Tavg = _pv(params, 'bubble_dMdt'), float('nan')

    Lgain = res.L_gain if res.L_gain is not None else float('nan')
    Lloss = res.L_loss if res.L_loss is not None else float('nan')
    ratio = ((Lgain - Lloss) / Lgain) if (np.isfinite(Lgain) and Lgain != 0) else float('nan')

    return {
        't_now': _pv(params, 't_now'),
        'cool_beta': float(res.beta), 'cool_delta': float(res.delta),
        'beta_plus_delta': float(res.beta) + float(res.delta),
        'Pb': _pv(params, 'Pb'), 'bubble_dMdt': dMdt,
        'Lmech_total': _pv(params, 'Lmech_total'),
        'Lmech_W': _pv(params, 'Lmech_W'), 'Lmech_SN': _pv(params, 'Lmech_SN'),
        'bubble_Lgain': Lgain, 'bubble_Lloss': Lloss, 'cooling_ratio': ratio,
        'v_struct_min': vmin, 'v_struct_nneg': nneg, 'v_struct_npts': npts,
        'v_neg_frac_thick': frac,
        'R2': _pv(params, 'R2'), 'v2': _pv(params, 'v2'), 'Eb': _pv(params, 'Eb'),
        'bubble_Tavg': Tavg, 'c_sound': _pv(params, 'c_sound'),
        'no_physical_root': bool(res.no_physical_root),
        'betadelta_converged': bool(res.converged),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('param')
    ap.add_argument('--out')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--hold-inflow', type=float, default=None,
                    help='Phase 6.1 counterfactual: when a segment inflow '
                         'thickness-fraction exceeds this, reject it and hold '
                         'the last physical structure (mimics no_physical_root).')
    ap.add_argument('--hold-after', type=float, default=None,
                    help='Positive control: hold (freeze) the structure on EVERY '
                         'segment after this t [Myr] -- a sustained perturbation '
                         'that must move the macro state if propagation works.')
    args = ap.parse_args()

    from trinity._input import read_param
    from trinity.cloud_properties.validate_gmc import validate_gmc_from_params

    params = read_param.read_param(args.param)

    chk = validate_gmc_from_params(params)
    print(f"[{args.param}] GMC valid={chk.valid}")
    for w in chk.warnings:
        print(f"  warn: {w}")
    for e in chk.errors:
        print(f"  ERROR: {e}")
    if args.validate_only or not chk.valid:
        sys.exit(0 if chk.valid else 1)

    from trinity import main as trinity_main
    import trinity.phase1b_energy_implicit.run_energy_implicit_phase as runmod

    # Stream rows to disk as they are computed (flush each), so progress
    # survives the ephemeral container being reclaimed mid-run -- a 45-min run
    # that dies at minute 40 still leaves every segment it reached on disk.
    fh = open(args.out, 'w', newline='') if args.out else None
    writer = csv.DictWriter(fh, fieldnames=FIELDS) if fh else None
    if writer:
        writer.writeheader()
        fh.flush()
    stats = {'n': 0, 'neg': 0, 'deep': 0, 'noroot': 0}
    orig = runmod.solve_betadelta_pure

    def wrapped(beta_guess, delta_guess, params, *a, **k):
        res = orig(beta_guess, delta_guess, params, *a, **k)
        # Phase 6.1 counterfactual: reject a structure and hold the last physical
        # one, by mimicking the runner's no_physical_root path (bubble_properties
        # =None -> updateDict skipped -> dMdt/structure held). Two triggers:
        #   --hold-inflow FRAC : the actual treatment (hold deep-inflow segments)
        #   --hold-after T0    : POSITIVE CONTROL -- hold EVERY segment after T0,
        #                        a large sustained perturbation that MUST move the
        #                        macro state if the channel propagates at all.
        t_now = _pv(params, 't_now')
        do_hold = False
        if args.hold_inflow is not None:
            f = _frac(res)
            if f is not None and f > args.hold_inflow:
                sys.stderr.write(f"HOLD inflow seg: frac={f:.2f} t={t_now:.3f}\n")
                do_hold = True
        if args.hold_after is not None and t_now is not None and t_now > args.hold_after:
            do_hold = True
        if do_hold:
            res = dataclasses.replace(
                res, bubble_properties=None, no_physical_root=True,
                L_gain=None, L_loss=None)
        try:
            row = _row(params, res)
            stats['n'] += 1
            if (row['v_struct_nneg'] or 0) > 0:
                stats['neg'] += 1
            if (row['v_struct_nneg'] or 0) >= 5:
                stats['deep'] += 1
            if row['no_physical_root']:
                stats['noroot'] += 1
            if writer:
                writer.writerow(row)
                fh.flush()
        except Exception as exc:  # never let diagnostics break the run
            sys.stderr.write(f"row capture failed: {exc}\n")
        return res

    runmod.solve_betadelta_pure = wrapped
    try:
        trinity_main.start_expansion(params)
    except SystemExit:
        pass
    finally:
        runmod.solve_betadelta_pure = orig
        if fh:
            fh.close()
        print(f"[{args.param}] wrote {stats['n']} rows -> {args.out} "
              f"(inflow={stats['neg']}, deep>=5pts={stats['deep']}, "
              f"no_root={stats['noroot']})")


if __name__ == '__main__':
    main()
