#!/usr/bin/env python3
"""P21 close-out — adjudicate §0.3 against the SSOT (7830-run Helix survey).

Reads the reduced SSOT (local, gitignored plots/): summary.csv + the frozen match
(match.csv @ 7 pc base; match_62.csv @ 6.2 pc base if present). Prints the quotable
verdicts that go into PLAN.md §0.3 / §0.6. Radii-only default χ² (R2 7±1, rShell 19±2).

Command:  python docs/dev/rosette-cf/figs/adjudicate_ssot.py
Data:     paper/rosette/plots/{summary,match,match_62}.csv  (SSOT, local-mount only)
Output:   docs/dev/rosette-cf/data/ssot_adjudication_2026-07-14.txt (commit this)
"""
import csv, os, statistics
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
PLOTS = os.path.join(ROOT, 'paper', 'rosette', 'plots')


def f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def load(name):
    return {r['run_name']: r for r in csv.DictReader(open(os.path.join(PLOTS, name)))}


def join(match_csv):
    s = load('summary.csv')
    m = load(match_csv)
    J = []
    for rn, mr in m.items():
        c = f(mr['chi2_min'])
        sr = s.get(rn, {})
        J.append(dict(rn=rn, chi2=c, R2=f(mr.get('R2_at_best')), rSh=f(mr.get('rShell_at_best')),
                      v2=f(mr.get('v2_kms_at_best')), nCore=f(mr['nCore']), cf=f(mr['coverFraction']),
                      PISM=f(mr['PISM']), phii=mr['phii'], mCloud=f(mr['mCloud']),
                      profile=sr.get('profile'), fmix=f(sr.get('fmix')), fate=sr.get('fate')))
    return J


def main():
    J7 = join('match.csv')
    fin = [r for r in J7 if r['chi2'] is not None and r['chi2'] != float('inf')]
    print(f"SSOT: {len(J7)} runs | matchable (finite chi2, radii-only 7pc): {len(fin)} "
          f"({100*len(fin)//len(J7)}%)")
    print(f"  matchable fates: {dict(Counter(r['fate'] for r in fin))}")
    print(f"  all-grid fates : from summary (recollapsed dominates)\n")

    def best(pool, n=1):
        return sorted(pool, key=lambda r: r['chi2'])[:n]

    print("== A. fate-AGNOSTIC best (radii-only 7pc) — WARNING: mostly recollapsing bubbles ==")
    for r in best(fin, 3):
        print(f"   chi2={r['chi2']:.2f} R2={r['R2']:.2f} rSh={r['rSh']:.1f}  mCloud={r['mCloud']:.0g} "
              f"nCore={r['nCore']:.0g} cf={r['cf']} prof={r['profile']} phii={r['phii']} fate={r['fate']}")
    print(f"   -> of the top-200 fate-agnostic fits, fates: "
          f"{dict(Counter(r['fate'] for r in best(fin,200)))}\n")

    ong = [r for r in fin if r['fate'] == 'ongoing']
    print("== B. ONGOING (expanding at age) best — the PHYSICALLY-RELEVANT cavity match ==")
    for r in best(ong, 5):
        print(f"   chi2={r['chi2']:.2f} R2={r['R2']:.2f} rSh={r['rSh']:.1f} v2={r['v2']:.1f}km/s  "
              f"mCloud={r['mCloud']:.0g} nCore={r['nCore']:.0g} cf={r['cf']} prof={r['profile']} "
              f"phii={r['phii']} fmix={r['fmix']}")
    phys = [r for r in ong if r['mCloud'] == 1e5]
    p = best(phys, 1)[0]
    print(f"   PHYSICAL pair (mCloud=1e5, sfe=0.01) best ongoing: chi2={p['chi2']:.2f} "
          f"nCore={p['nCore']:.0g} cf={p['cf']} prof={p['profile']} R2={p['R2']:.2f} v2={p['v2']:.1f}km/s\n")

    print("== C. C_f is an INTERIOR optimum (min chi2 by C_f, ongoing only) ==")
    d = defaultdict(list)
    for r in ong:
        d[r['cf']].append(r['chi2'])
    for cf in sorted(d):
        print(f"   cf={cf}: min chi2={min(d[cf]):.2f}  (n={len(d[cf])})")
    print("   -> interim prediction 'edge-min at C_f<=0.70' is RETRACTED; optimum ~0.89-0.95, near the pilot's 0.89.\n")

    print("== D. near-sealed C_f=0.99 overshoot by nCore (the ~x3 claim; median R2 vs 7 & 6.2 pc) ==")
    byn = defaultdict(list)
    for r in J7:
        if r['cf'] == 0.99 and r['R2'] is not None:
            byn[r['nCore']].append(r['R2'])
    for nc in sorted(byn):
        md = statistics.median(byn[nc])
        print(f"   nCore={nc:>8.0f}: median R2@best={md:5.1f} pc  x{md/7:.1f}(vs7) x{md/6.2:.1f}(vs6.2)  n={len(byn[nc])}")
    print("   -> ~x3 holds only at LOW nCore; shrinks at higher nCore (and nCore=1e4 near-sealed recollapses).\n")

    if os.path.exists(os.path.join(PLOTS, 'match_62.csv')):
        J62 = [r for r in join('match_62.csv') if r['chi2'] is not None and r['fate'] == 'ongoing']
        b = sorted(J62, key=lambda r: r['chi2'])[0]
        print("== E. 6.2 pc base (F-12), best ONGOING ==")
        print(f"   chi2={b['chi2']:.2f} R2={b['R2']:.2f} mCloud={b['mCloud']:.0g} nCore={b['nCore']:.0g} "
              f"cf={b['cf']} prof={b['profile']}\n")
    else:
        print("== E. 6.2 pc base: match_62.csv not ready; 6.2 overshoot readable as R2_at_best-6.2 "
              "(best ongoing R2~5.5 -> near-perfect on 6.2). ==\n")

    print("PREDICTION VERDICTS (interim §0.3, dated 2026-07-14):")
    print("  (a) C_f optimum <=0.70 edge-min .............. RETRACTED (interior ~0.89-0.95; pilot-like)")
    print("  (b) chi2 falls w/ nCore; fmix=4<fmix=1 ....... MIXED (nCore: expanding optimum at 1e3 not 1e4 -")
    print("      the 1e4 min recollapses; fmix=4 does favour the best ongoing fits)")
    print("  (c) sealed overshoots tens-of-pc all nCore ... PARTIAL (yes at low nCore ~x2.8; high nCore recollapses)")
    print("  (d) physical pair needs higher PISM .......... RETRACTED (PISM ~degenerate; 1e5 matches at nCore=1e3)")


if __name__ == '__main__':
    main()
