"""Phase 3: audit production against the Phase-2-validated ground truth.

For each dumped state, run BOTH:
  * production  -- bl.get_bubbleproperties_pure (the real grid + trapezoid path)
  * reference   -- the Phase-2-validated, converged, method-independent ground
                   truth (reference_bubble_luminosity)
on IDENTICAL structure inputs (production re-solves R1/Pb/dMdt to the dumped
values, rel 0 -- verified), so the comparison isolates the INTEGRATION method.

Each of the 7 quantities is classified against the ground truth:
  match        rel < 1e-4
  minor        1e-4 <= rel < 1e-2   (small grid/quadrature error)
  SIGNIFICANT  rel >= 1e-2          (production deviates -- flag)

This is the decisive measurement: is the current solver correct, and where?
The reference is judged correct on its OWN terms (Phase 2), so this is not
circular. No production change.
"""
from __future__ import annotations

import os
import sys
import glob


sys.path.insert(0, os.path.dirname(__file__))
from audit import load_state  # noqa: E402
from reference import reference_bubble_luminosity  # noqa: E402
import trinity.bubble_structure.bubble_luminosity as bl  # noqa: E402

# production BubbleProperties field  ->  reference key
PROD_TO_REF = {
    'bubble_L1Bubble': 'L1Bubble',
    'bubble_L2Conduction': 'L2Conduction',
    'bubble_L3Intermediate': 'L3Intermediate',
    'bubble_LTotal': 'L_total',
    'bubble_mass': 'mBubble',
    'bubble_Tavg': 'Tavg',
    'bubble_T_r_Tb': 'T_rgoal',
}
MATCH, MINOR, SIG = 1e-4, 1e-2, 1e-2


def _classify(rel):
    if rel < MATCH:
        return "match"
    if rel < SIG:
        return "minor"
    return "SIGNIFICANT"


def audit_state(params, inputs):
    a = inputs
    props = bl.get_bubbleproperties_pure(params)            # production
    ref = reference_bubble_luminosity(                      # ground truth
        params, a['R1'], a['Pb'], a['r2Prime'],
        a['initial_conditions'], a['bubble_r_Tb'])
    rows = {}
    for pk, rk in PROD_TO_REF.items():
        pv = float(getattr(props, pk))
        rv = float(ref[rk])
        rel = abs(pv - rv) / max(abs(rv), 1e-300)
        rows[rk] = (pv, rv, rel, _classify(rel))
    return rows


def main(argv):
    if not argv:
        print("usage: python production_audit.py <states_dir|state.pkl> [base.param]")
        return 2
    target, base = argv[0], (argv[1] if len(argv) > 1 else None)
    files = sorted(glob.glob(os.path.join(target, "*.pkl"))) if os.path.isdir(target) else [target]
    worst_per_qty = {}
    any_sig = False
    for f in files:
        kwargs = {} if base is None else {'base_param': base}
        params, inputs, ref_arrays, meta = load_state(f, **kwargs)
        print(f"\n================ {os.path.basename(f)} ================")
        print(f"  {'quantity':16} {'production':>14} {'reference(GT)':>14} {'rel':>10}  verdict")
        try:
            rows = audit_state(params, inputs)
        except Exception as e:
            print(f"  AUDIT FAILED: {type(e).__name__}: {e}")
            any_sig = True
            continue
        for rk, (pv, rv, rel, verdict) in rows.items():
            worst_per_qty[rk] = max(worst_per_qty.get(rk, 0.0), rel)
            if verdict == "SIGNIFICANT":
                any_sig = True
            print(f"  {rk:16} {pv:14.6e} {rv:14.6e} {rel:10.2e}  {verdict}")
    print("\n================ summary: worst rel diff per quantity ================")
    for rk in PROD_TO_REF.values():
        w = worst_per_qty.get(rk, float('nan'))
        print(f"  {rk:16} worst_rel={w:.2e}  {_classify(w)}")
    print(f"\n=== Phase-3 audit: {'production deviates somewhere (see SIGNIFICANT)' if any_sig else 'production within minor tolerance of ground truth everywhere'} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
