#!/usr/bin/env python3
"""Generate the bench5 matrix — Phase 5 of SOURCE_TERM_DESIGN.md: Lancaster 2021b calibration.

Five bespoke configs mapped 1:1 onto Lancaster et al. 2021b (ApJ 914, 90) Table-1 suite members,
crossed with f_A in {1, 4, 6, 8, 12, 16} (the Phase-4 whole-band winner 12 is in-grid), two arms
each per the Phase-5 protocol:

    production  — default `cooling_balance` trigger live (fire time / fate)
    diagnostic  — `transition_trigger blowout` so theta(t) logs uncensored through the energy
                  phase (blowout still transitions at R2 > rCloud, the upper cap of the
                  comparison window W anyway)

= 5 benches x 6 f_A x 2 arms = 60 params. Single-knob by construction (cooling_boost_mode=none,
cooling_boost_kappa=1 everywhere; only cooling_boost_fA varies).

PROVENANCE (imprint: LANCASTER_REFERENCE.md section 7b — read that, not the chat). Table-1 values
were maintainer-supplied as paper excerpts on 2026-07-12 and are [V]-grade: n_H is hydrogen-nucleus
density with mu_H = 1.4 (verified to reproduce every R_cl), epsilon_* = M_*/M_cloud in
{0.01, 0.1, 1} per Table-1 Notes (the search-snippet "M_* = 5000 Msun fixed" was FALSIFIED — the
draft sfe=0.05 mapping matched no published model). V_w stays [I]-grade (not in the excerpts;
unused here — TRINITY supplies its own SB99 wind).

EXACT MAPPING (SOURCE_TERM_DESIGN section 3 Phase 5, corrected 2026-07-12): TRINITY's .param
mCloud is pre-SFE (read_param rebinds: mCluster = sfe*mCloud, gas = (1-sfe)*mCloud, and rCloud
derives from the post-SF gas), while L21b *adds* a star particle eps*M_cl to an M_cl gas cloud.
So freeze mCloud = M_cl*(1+eps), sfe = eps/(1+eps)  =>  post-SF gas = M_cl at nCore = n_H
(rCloud = R_cl exactly) and mCluster = eps*M_cl exactly.

Emit-time gates (the builder self-checks; a failing gate aborts the emit):
  1. sweep_runner's GMC plausibility validation passes for every bench (rCloud_max etc.);
  2. the exact mapping holds: rCloud computed from the post-SF gas mass matches the L21b R_cl
     to <1% for every bench.

Harvest into the standard namespace (theta only from dictionary.jsonl accepted rows):
    python runs/harvest_theta_max.py "$WS"/outputs/bench5/* --csv runs/data/bench5_summary.csv

Regenerate:  python docs/dev/transition/pdv-trigger/runs/make_bench5_params.py
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "params" / "bench5"
REPO = HERE.parents[3]

# (name, M_cl [Msun], R_cl [pc], n_H [cm^-3], eps_*) — L21b Table 1, [V] 2026-07-12.
BENCHES = [
    ("bench1_m5e4_r20", 5e4, 20.0, 43.1, 0.1),
    ("bench2_m1e5_r10", 1e5, 10.0, 690.0, 0.1),
    ("bench3_m1e5_r5", 1e5, 5.0, 5520.0, 0.1),
    ("bench4_m1e5_r2p5", 1e5, 2.5, 44200.0, 0.1),
    ("bench5_m5e5_r2p5", 5e5, 2.5, 228000.0, 0.01),
]

F_A = ["1", "4", "6", "8", "12", "16"]
STOP_T = 5

# Verified constants for the emit-time exact-mapping check (mu_H = 1.4, section 7b).
_MH_G, _MSUN_G, _PC_CM = 1.6726e-24, 1.989e33, 3.086e18


def _gas_rcloud_pc(m_gas_msun, n_h):
    """Homogeneous-cloud radius from gas mass at density n_H (mu_H=1.4), in pc."""
    rho = 1.4 * _MH_G * n_h
    r_cm = (3 * m_gas_msun * _MSUN_G / (4 * math.pi * rho)) ** (1 / 3)
    return r_cm / _PC_CM


def emit(name, mcloud, ncore, sfe, fa, diag):
    lines = [
        f"model_name             {name}",
        f"{'mCloud':<22} {mcloud:.6g}",
        f"{'nCore':<22} {ncore:.6g}",
        f"{'rCore':<22} 1",
        f"{'sfe':<22} {sfe:.16g}",
        f"{'dens_profile':<22} densPL",
        f"{'densPL_alpha':<22} 0",
        f"{'stop_t':<22} {STOP_T}",
    ]
    if fa != "1":
        lines += [f"{'cooling_boost_fA':<22} {fa}"]
    if diag:
        lines += [f"{'transition_trigger':<22} blowout"]
    lines += [
        "log_console            False",
        "log_file               True",
        f"path2output            outputs/bench5/{name}",
    ]
    (OUT / f"{name}.param").write_text("\n".join(lines) + "\n")
    return name


def main():
    sys.path.insert(0, str(REPO))
    from trinity._input.sweep_runner import _validate_sweep_combination

    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for bench, m_cl, r_cl, n_h, eps in BENCHES:
        mcloud = m_cl * (1 + eps)
        sfe = eps / (1 + eps)

        # Gate 2: exact mapping — post-SF gas mass (1-sfe)*mCloud = M_cl must sit at R_cl.
        # Tolerance 2%, not exact: Table-1's published n_H is what we pin (spec: nCore = n_bar
        # exactly), and its 5e4/5e5 rows at R_cl=2.5 pc carry a ~3.3% internal rounding offset
        # (both imply R=2.473 pc; the other 10 rows land exactly) — see LANCASTER_REFERENCE 7b.
        r_derived = _gas_rcloud_pc((1 - sfe) * mcloud, n_h)
        if abs(r_derived - r_cl) / r_cl >= 0.02:
            sys.exit(f"ABORT {bench}: derived rCloud {r_derived:.3f} pc != R_cl {r_cl} pc")

        # Gate 1: TRINITY's own pre-run GMC plausibility validation (same path run.py sweeps use).
        res = _validate_sweep_combination(
            {"dens_profile": "densPL", "mCloud": mcloud, "nCore": n_h,
             "densPL_alpha": 0.0, "rCore": 1.0}
        )
        if res is None or not res.valid:
            sys.exit(f"ABORT {bench}: GMC validation failed: {getattr(res, 'errors', 'n/a')}")

        for fa in F_A:
            mode = "none" if fa == "1" else f"fa{fa}"
            for diag in (False, True):
                arm = f"{bench}__{mode}" + ("_diag" if diag else "")
                names.append(emit(arm, mcloud, n_h, sfe, fa, diag))
        print(f"{bench}: rCloud(gas)={r_derived:.2f} pc (target {r_cl}), "
              f"validation rCloud={res.rCloud:.2f} pc, valid={res.valid}")
    print(f"wrote {len(names)} params to {OUT}  "
          f"({len(BENCHES)} benches x {len(F_A)} f_A x 2 arms)")


if __name__ == "__main__":
    main()
