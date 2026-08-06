"""Budget closure: do the recorded parts sum to the recorded totals?

The cheapest test for a dropped or double-counted term, and it needs no new run —
every quantity is already in dictionary.jsonl. A decomposition that fails here is a
term the code computed one way and reported another.

    python docs/dev/code-audit/harness/phase6_budget.py outputs/<run>/dictionary.jsonl

Reports; does not assert. A "violation" is a relative mismatch above TOL at a
snapshot. Identities are only checked where their denominator is non-zero, so a
quantity that is legitimately zero in a phase does not manufacture failures.
"""

import json
import math
import sys

TOL = 1e-9


def identities(r):
    """(name, lhs, rhs) triples that must hold snapshot-by-snapshot."""
    area = 4 * math.pi * r["R2"] ** 2
    return [
        (
            "bubble_LTotal == L1 + L2 + L3",
            r["bubble_LTotal"],
            r["bubble_L1Bubble"] + r["bubble_L2Conduction"] + r["bubble_L3Intermediate"],
        ),
        ("Lmech_total == Lmech_W + Lmech_SN", r["Lmech_total"], r["Lmech_W"] + r["Lmech_SN"]),
        (
            "pdot_total == pdot_W + pdot_SN",
            r["pdot_total"],
            r.get("pdot_W", 0) + r.get("pdot_SN", 0),
        ),
        ("F_HII == P_HII * 4pi R2^2", r["F_HII"], r["P_HII"] * area),
        ("F_ion_in == press_HII_in * 4pi R2^2", r["F_ion_in"], r["press_HII_in"] * area),
        # The S6-R-02 refutation established 4pi R2^2 * pRam == pdot_total
        # analytically; this is the same identity measured on a real trajectory.
        ("F_ram_wind == pdot_total", r["F_ram_wind"], r["pdot_total"]),
        # F_ram is *named* for the ram-pressure force, so it should be the recorded
        # ram pressure times the area. It is not — see P6-06.
        ("F_ram == P_ram * 4pi R2^2", r["F_ram"], r["P_ram"] * area),
    ]


def main(path):
    with open(path) as fh:
        rows = [json.loads(line) for line in fh]

    print(f"# budget closure — {path}\n")
    print(f"{len(rows)} snapshots, tolerance {TOL:g} relative\n")
    print("| identity | violations | max rel | verdict |")
    print("|---|---:|---:|---|")

    names = [n for n, _, _ in identities(rows[0])]
    for i, name in enumerate(names):
        worst, bad, checked = 0.0, 0, 0
        for r in rows:
            try:
                _, lhs, rhs = identities(r)[i]
            except KeyError:
                continue
            if rhs == 0 and lhs == 0:
                continue  # both sides trivially zero: no information
            checked += 1
            denom = max(abs(rhs), abs(lhs))
            rel = abs(lhs - rhs) / denom if denom else 0.0
            worst = max(worst, rel)
            bad += rel > TOL
        if not checked:
            print(f"| `{name}` | — | — | not exercised (both sides zero) |")
            continue
        verdict = "**CLOSES**" if bad == 0 else f"**FAILS** ({bad}/{checked})"
        print(f"| `{name}` | {bad}/{checked} | {worst:.3e} | {verdict} |")


if __name__ == "__main__":
    main(sys.argv[1])
