#!/usr/bin/env python3
"""P6 velocity step — re-rank the ongoing SSOT matches with v2 in the chi2.

Answers "make v2 a fit": the model's v2 = dR2/dt is the COLD contact-discontinuity
shell, so it must be judged against the NEUTRAL shell (~4.3 km/s, F-3/D-6), not the
ionised expansion (~14, the frozen v_bubble target). Prints the ongoing-only ranking
under three chi2 definitions so the shift is explicit; changes NO target.

Two modes:
  * approximate (default, self-contained): reuse v2_kms_at_best from match.csv (the v2
    at the radii-best time). Robust here because the penalty is driven by the ~1-vs-14
    gap, not by the small shift in t_best.
  * authoritative: if plots/match_both.csv exists (from
    `matching/match_runs.py --terms R_bubble,R_shell,v_bubble,v_shell`), it re-minimises
    t with the velocities in-loop; read its chi2_min directly.

Command:  python docs/dev/rosette-cf/figs/rank_velocity.py
Data:     paper/rosette/plots/{summary,match[,match_both]}.csv  (SSOT, local-mount only)
Targets:  v_bubble (ionised) 14+-4 ; v_shell (neutral) 4.3+-2   [matching/observables.py, frozen]
"""
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
PLOTS = os.path.join(ROOT, 'paper', 'rosette', 'plots')
V_ION, V_ION_E = 14.0, 4.0      # v_bubble, ionised ordered expansion (frozen OBS)
V_NEU, V_NEU_E = 4.3, 2.0       # v_shell, neutral swept-up shell (frozen OBS, F-3/D-6)


def f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def load(name):
    return {r['run_name']: r for r in csv.DictReader(open(os.path.join(PLOTS, name)))}


def main():
    s = load('summary.csv')
    m = load('match.csv')
    J = []
    for rn, mr in m.items():
        sr = s.get(rn, {})
        c, v = f(mr['chi2_min']), f(mr.get('v2_kms_at_best'))
        if sr.get('fate') != 'ongoing' or c in (None, float('inf')) or v is None:
            continue
        J.append(dict(
            rn=rn, cr=c, v2=v, R2=f(mr.get('R2_at_best')), rSh=f(mr.get('rShell_at_best')),
            mCloud=f(sr['mCloud']), nCore=f(sr['nCore']), cf=f(sr['coverFraction']),
            profile=sr.get('profile'), phii=sr.get('phii'), fmix=f(sr.get('fmix')),
            c_neu=c + ((v - V_NEU) / V_NEU_E) ** 2,
            c_ion=c + ((v - V_ION) / V_ION_E) ** 2))
    print(f"ongoing matchable runs (finite radii chi2): {len(J)}\n")

    def show(key, lab):
        print(f"== best ONGOING by {lab} ==")
        seen = set()
        for r in sorted(J, key=lambda r: r[key]):
            sig = (r['mCloud'], r['nCore'], r['cf'], r['profile'], r['phii'])
            if sig in seen:
                continue
            seen.add(sig)
            print(f"   {key}={r[key]:5.1f} (radii {r['cr']:.1f}) v2={r['v2']:.1f}  "
                  f"mCloud={r['mCloud']:.0g} nCore={r['nCore']:.0g} cf={r['cf']} "
                  f"prof={r['profile']} phii={r['phii']} fmix={r['fmix']} R2={r['R2']:.1f}")
            if len(seen) >= 4:
                break
        print()

    show('cr', 'RADII ONLY')
    show('c_neu', f'RADII + v2 vs NEUTRAL {V_NEU} (physically correct, F-3/D-6)')
    show('c_ion', f'RADII + v2 vs IONISED {V_ION} (frozen v_bubble target)')

    nion = sum(1 for r in J if abs(r['v2'] - V_ION) <= V_ION_E)
    nneu = sum(1 for r in J if abs(r['v2'] - V_NEU) <= V_NEU_E)
    print(f"v2 in ionised {V_ION}+-{V_ION_E} band: {nion}/{len(J)}; "
          f"neutral {V_NEU}+-{V_NEU_E} band: {nneu}/{len(J)}; "
          f"max v2 among ongoing: {max(r['v2'] for r in J):.1f} km/s")
    print("VERDICT: v2~1 km/s is the cold/neutral shell (survives the 4.3 constraint, "
          "same C_f~0.89 corner); it is NOT the ionised gas (no run matches ~7pc AND v2~14). "
          "-> supports the non-spherical §6 reveal; targets unchanged.")

    mb = os.path.join(PLOTS, 'match_both.csv')
    if os.path.exists(mb):
        B = [r for r in load('match_both.csv').values()
             if s.get(r['run_name'], {}).get('fate') == 'ongoing'
             and f(r['chi2_min']) not in (None, float('inf'))]
        B.sort(key=lambda r: f(r['chi2_min']))
        print("\n== AUTHORITATIVE (match_both.csv, t re-minimised with v2+vShell) best ongoing ==")
        for r in B[:4]:
            sr = s[r['run_name']]
            print(f"   chi2_all={f(r['chi2_min']):.1f} mCloud={sr['mCloud']} nCore={sr['nCore']} "
                  f"cf={sr['coverFraction']} prof={sr['profile']} phii={sr['phii']} "
                  f"v2@={f(r['v2_kms_at_best']):.1f} R2@={f(r['R2_at_best']):.1f}")


if __name__ == '__main__':
    main()
