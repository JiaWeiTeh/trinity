"""P-sens (offline) on the committed transition CSVs — NO new runs.

Family-agnostic P-sens items from TRANSITION_TRIGGER_PLAN.md (F2-specific items
1-2 skipped: F2 eliminated at G0):
  item 5 -- epsilon sensitivity of the F0 firing epoch (flat configs that cross),
  item 4 -- sustained-over-t_cross rule vs instantaneous F0,
  plus the structural test: does the transition (cooling OR blowout) fire BEFORE
  the first WR/SN surge (beta<0)? If so the reset pathology is moot in practice.

Usage: python scratch/transition/psens.py
"""
import csv

CONFIGS = ["mock_hybr", "dense_flat", "steep", "steep_long"]  # legacy mock4e3 = clamp artifact, skip
EPS = [0.02, 0.03, 0.05, 0.07, 0.10]


def load(name):
    rows = list(csv.DictReader(open(f"analysis/data/transition_{name}.csv")))
    f = lambda k: [float(r[k]) for r in rows]
    return dict(t=f("t_now"), ratio=f("ratio_F0"), R2=f("R2"), cs=f("c_sound"),
                beta=f("cool_beta"), rrc=f("R2_over_rCloud"))


def first_below(t, x, thr):
    return next((ti for ti, xi in zip(t, x) if xi < thr), None)


def first_above(t, x, thr):
    return next((ti for ti, xi in zip(t, x) if xi > thr), None)


def first_sustained_below(t, x, thr, tcross):
    """Fire at first t where x<thr has held continuously for >= local t_cross."""
    start = None
    for ti, xi, tc in zip(t, x, tcross):
        if xi < thr:
            if start is None:
                start = ti
            if ti - start >= tc:
                return ti
        else:
            start = None
    return None


def fmt(x):
    return f"{x:.4f}" if x is not None else "—"


print("=== item 5: F0 firing epoch vs epsilon (Myr) ===")
print(f"{'config':>11} " + "".join(f"{'e='+str(e):>9}" for e in EPS))
for n in CONFIGS:
    d = load(n)
    print(f"{n:>11} " + "".join(f"{fmt(first_below(d['t'], d['ratio'], e)):>9}" for e in EPS))

print("\n=== item 4: instantaneous vs sustained-over-t_cross F0 (eps=0.05) ===")
print(f"{'config':>11} {'F0 inst':>9} {'F0 sustained':>13} {'t_cross@trans':>14}")
for n in CONFIGS:
    d = load(n)
    tcross = [r / c if c > 0 else 0.0 for r, c in zip(d["R2"], d["cs"])]
    inst = first_below(d["t"], d["ratio"], 0.05)
    sus = first_sustained_below(d["t"], d["ratio"], 0.05, tcross)
    tcx = tcross[d["t"].index(inst)] if inst is not None else float("nan")
    print(f"{n:>11} {fmt(inst):>9} {fmt(sus):>13} {tcx:>14.4f}")

print("\n=== structural: does transition (cooling OR blowout) precede the 1st WR surge? ===")
print(f"{'config':>11} {'F0(0.05)':>9} {'blowout F4':>11} {'1st beta<0':>11} {'transition':>11} {'precedes?':>10}")
for n in CONFIGS:
    d = load(n)
    f0 = first_below(d["t"], d["ratio"], 0.05)
    f4 = first_above(d["t"], d["rrc"], 1.0)
    surge = next((ti for ti, b in zip(d["t"], d["beta"]) if b < 0), None)
    trans = min([x for x in (f0, f4) if x is not None], default=None)
    if trans is None:
        verdict = "no-trans"
    elif surge is None or trans < surge:
        verdict = "YES"
    else:
        verdict = "NO"
    print(f"{n:>11} {fmt(f0):>9} {fmt(f4):>11} {fmt(surge):>11} {fmt(trans):>11} {verdict:>10}")
