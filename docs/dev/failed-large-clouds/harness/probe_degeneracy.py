#!/usr/bin/env python3
"""Sim-free probe of the R1->R2 catastrophic-cooling degeneracy.

Characterises how the inner wind shock R1 (solve_R1) and the bubble pressure
Pb (bubble_E2P) behave as the bubble energy Eb collapses, for the failing
large-cloud crash state (R2=7.0475 pc, Lmech=5.07e12, v_mech=3739; code units).

It also evaluates a candidate fix -- clamping R1 <= R2*(1-eps) -- so we can see
the resulting (finite) Pb and judge whether it stays physical.

Instant (no integration). Run:
    python docs/dev/failed-large-clouds/harness/probe_degeneracy.py

Writes docs/dev/failed-large-clouds/data/probe_degeneracy.csv (committed artifact).
"""
import csv
import os
import numpy as np

import trinity.bubble_structure.get_bubbleParams as gbp
import trinity._functions.unit_conversions as cvt

# Crash state pulled from the real Helix log (5e9, n1e2) and the local repro.
R2 = 7.047540          # pc
LMECH = 5.070535e12    # code units
VMECH = 3.739310e3     # pc/Myr
GAMMA = 5.0 / 3.0

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "data", "probe_degeneracy.csv")


def clamped_E2P(Eb, R2, R1, gamma, rel_floor=1e-6):
    """Candidate V1: floor the shell volume (R2^3 - R1^3) at a relative epsilon
    of R2^3 so the denominator can never be zero. Mirrors bubble_E2P's cgs math.
    """
    r1 = R1 * cvt.pc2cm
    r2 = R2 * cvt.pc2cm
    Eb_cgs = Eb * cvt.E_au2cgs
    vol = r2**3 - r1**3
    floor = rel_floor * r2**3
    vol = max(vol, floor)
    Pb = (gamma - 1) * Eb_cgs / vol / (4 * np.pi / 3)
    return Pb * cvt.Pb_cgs2au


def main():
    rows = []
    # Sweep Eb from a healthy value down through the collapse.
    for Eb in np.logspace(9, -8, 18):
        R1 = gbp.solve_R1(R2, Eb, LMECH, VMECH)
        gap = R2 - R1
        vol_pc = R2**3 - R1**3                # pc^3, the bubble_E2P denominator basis
        rel_vol = vol_pc / R2**3
        # baseline bubble_E2P (may overflow -> inf, or raise ZeroDivisionError)
        try:
            Pb_base = gbp.bubble_E2P(Eb, R2, R1, GAMMA)
            base_note = "ok"
        except ZeroDivisionError:
            Pb_base = float("inf")
            base_note = "ZeroDivisionError"
        Pb_clamp = clamped_E2P(Eb, R2, R1, GAMMA)
        rows.append({
            "Eb": f"{Eb:.3e}", "R1": f"{R1:.6f}", "R2_minus_R1": f"{gap:.3e}",
            "rel_shell_vol": f"{rel_vol:.3e}",
            "Pb_base": f"{Pb_base:.3e}", "base_note": base_note,
            "Pb_clamp_relfloor1e-6": f"{Pb_clamp:.3e}",
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Console summary
    print(f"R2={R2} pc, Lmech={LMECH:.3e}, v_mech={VMECH:.3e}")
    print(f"{'Eb':>10} {'R1':>10} {'R2-R1':>11} {'rel_vol':>11} {'Pb_base':>12} {'note':>16} {'Pb_clamp':>12}")
    for r in rows:
        print(f"{r['Eb']:>10} {r['R1']:>10} {r['R2_minus_R1']:>11} {r['rel_shell_vol']:>11} "
              f"{r['Pb_base']:>12} {r['base_note']:>16} {r['Pb_clamp_relfloor1e-6']:>12}")
    print(f"\nwrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
